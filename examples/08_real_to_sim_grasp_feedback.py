#!/usr/bin/env python3
"""示例 8：真实机械臂重力补偿 + MuJoCo 抓取方块 + 夹爪力反馈。

功能：
    1. 真实 B601-RS 机械臂进入 MIT 重力补偿模式，可自由拖动。
    2. 真实臂/夹爪状态实时同步到 MuJoCo 数字孪生。
    3. MuJoCo 场景中的方块可被真实夹爪抓取。
    4. 当夹爪与方块接触时，接触力会转换为反馈力矩施加到真实夹爪电机上。

主要参数：
    --mock              使用模拟数据，不连接真实硬件。
    --headless          不打开 MuJoCo 可视化窗口。
    --can can0          CAN 接口名（默认 can0）。
    --hw-yaml FILE      SDK 硬件配置文件（默认 rebotarm_rs.yaml）。
    --rate HZ           控制/同步频率（默认 50）。
    --no-hold           使能后电机不保持位置，方便手动拖动。

臂重力补偿：
    --kp-arm 8.0        臂关节 MIT kp。
    --kd-arm 1.0        臂关节 MIT kd。
    --gravity-scale     重力补偿缩放。支持单个值或 6 个逗号分隔值，
                        例如 --gravity-scale 1.0,1.2,1.0,1.0,1.0,1.0。
                        也可以在文件顶部直接修改 GRAVITY_SCALE 数组。
    --tau-arm-limit 20.0  臂前馈力矩上限（N·m）。

夹爪力反馈：
    --kp-gripper 0.0    夹爪 MIT kp。
    --kd-gripper 0.05   夹爪 MIT kd。保持较小，夹爪才能用手轻松打开。
    --force-scale 0.05  MuJoCo 接触力（N）到真实夹爪力矩（N·m）的缩放。
                        越大，抓到方块时夹得越紧。
    --tau-gripper-limit 1.0  夹爪反馈力矩上限（N·m）。
    --gripper-scale 0.00833  真实夹爪电机位置（rad）到 MuJoCo 直线位移（m）的缩放。

用法示例：
    # 连接真实机械臂（需先启动 CAN 接口）
    python examples/08_real_to_sim_grasp_feedback.py

    # 模拟模式（无硬件）
    python examples/08_real_to_sim_grasp_feedback.py --mock

    # 无窗口运行
    python examples/08_real_to_sim_grasp_feedback.py --headless

    # 给下坠关节加大重力补偿
    python examples/08_real_to_sim_grasp_feedback.py --gravity-scale 1.0,1.2,1.0,1.0,1.0,1.0

真实机械臂连接前请确保：
    sudo ip link set can0 up type can bitrate 500000

交互提示：
    - 在真实模式下，用手拖动机械臂使夹爪靠近并闭合在方块上。
    - 当 MuJoCo 中夹爪触碰到方块时，终端会显示接触力与反馈力矩。
    - 夹爪默认 kd 很小（0.05），所以空手可以轻松掰开；
      碰到方块后 force-scale 会生成反馈力矩，使其夹紧。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rebot_b601_rs_sim.bridge.grasp_feedback import GraspFeedback
from rebot_b601_rs_sim.bridge.real_to_sim import (
    MockRealRobot,
    RealToSimBridge,
    create_real_arm,
)
from rebot_b601_rs_sim.control.gravity_compensation import GravityCompensator
from rebot_b601_rs_sim.robot.model import RobotModel

# 按关节 1~6 的重力补偿缩放系数。
# 如果某个关节下坠，直接把对应位置的数值改大即可，例如：
# GRAVITY_SCALE = np.array([1.0, 1.2, 1.0, 1.0, 1.0, 1.0])
GRAVITY_SCALE = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])


def _limit_array(arr: np.ndarray, limit: float | np.ndarray) -> np.ndarray:
    """将数组裁剪到 ±limit 范围内。"""
    arr = np.asarray(arr, dtype=float)
    limit = np.asarray(limit, dtype=float)
    return np.clip(arr, -limit, limit)


def _parse_gravity_scale(value: str) -> np.ndarray:
    """解析重力补偿缩放系数。

    支持两种形式：
      - 单个值，如 "1.2"，表示 6 个臂关节都用 1.2
      - 逗号分隔的 6 个值，如 "1.0,1.2,1.0,1.0,1.0,1.0"，分别对应 joint1~joint6
    """
    parts = [p.strip() for p in value.split(",")]
    if len(parts) == 1:
        scale = float(parts[0])
        return np.full(6, scale)
    if len(parts) == 6:
        return np.array([float(p) for p in parts], dtype=float)
    raise argparse.ArgumentTypeError(
        f"gravity-scale must be a single float or 6 comma-separated floats, got {value!r}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-to-sim grasp with gravity compensation and gripper force feedback."
    )
    parser.add_argument("--mock", action="store_true", help="Use mock data instead of real hardware")
    parser.add_argument("--headless", action="store_true", help="Run without MuJoCo viewer window")
    parser.add_argument("--can", type=str, default="can0", help="CAN interface name (default: can0)")
    parser.add_argument(
        "--hw-yaml",
        type=str,
        default="rebotarm_rs.yaml",
        help="SDK hardware configuration YAML (default: rebotarm_rs.yaml)",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=50.0,
        help="Control and sync rate in Hz (default: 50)",
    )
    parser.add_argument(
        "--no-hold",
        action="store_true",
        help="Disable motor holding torque after enable (allows manual dragging)",
    )
    parser.add_argument(
        "--gripper-scale",
        type=float,
        default=0.05 / 6.021,
        help="Scale from real gripper motor position (rad) to MuJoCo slide displacement (m). "
             "With joint_left/joint_right range 0~0.05 m and real gripper 0~345deg, "
             "scale = 0.05 / 6.021 = 0.00830. Default: 0.00830",
    )
    parser.add_argument(
        "--gripper-offset",
        type=float,
        default=0.0,
        help="Real gripper motor angle (rad) that corresponds to fully closed (disp=0). "
             "Default: 0.0 rad (0deg)",
    )
    parser.add_argument(
        "--force-scale",
        type=float,
        default=0.05,
        help="Scale from MuJoCo contact force (N) to real gripper feedback torque (N·m). "
             "Tune on real hardware. Default: 0.05",
    )
    parser.add_argument("--kp-arm", type=float, default=0.0, help="Arm MIT kp (default: 0.0). Set to 0 for pure gravity compensation so the arm can be dragged.")
    parser.add_argument("--kd-arm", type=float, default=0.2, help="Arm MIT kd (default: 0.2). Small damping for stability while dragging.")
    parser.add_argument(
        "--gravity-scale",
        type=_parse_gravity_scale,
        default=GRAVITY_SCALE.copy(),
        help="Per-joint scale on gravity compensation torque. "
             "Either a single float applied to all 6 arm joints, "
             "or 6 comma-separated floats for joint1~joint6. "
             "Default is the GRAVITY_SCALE array defined at the top of this file. "
             "Example: --gravity-scale 1.0,1.2,1.0,1.0,1.0,1.0",
    )
    parser.add_argument("--kp-gripper", type=float, default=0.0, help="Gripper MIT kp (default: 0.0)")
    parser.add_argument(
        "--kd-gripper",
        type=float,
        default=0.05,
        help="Gripper MIT kd (default: 0.05). "
             "Keep small so the gripper can be opened easily by hand.",
    )
    parser.add_argument(
        "--tau-arm-limit",
        type=float,
        default=20.0,
        help="Arm feedforward torque limit (N·m, default: 20.0)",
    )
    parser.add_argument(
        "--tau-gripper-limit",
        type=float,
        default=1.0,
        help="Gripper feedback torque limit (N·m, default: 1.0). "
             "Caps the contact-reflex torque while grasping the cube.",
    )
    args = parser.parse_args()

    print("[08_grasp_feedback] Starting...", flush=True)

    robot = RobotModel()

    if args.mock:
        print("[mock mode] No real hardware connected.")
        # mock 模式下包含 gripper，便于测试夹爪反馈
        # 初始位姿让夹爪位于方块正上方可抓取位置
        # 真实 gripper 0°闭合、345°张开；mock 用 220° 对应单侧约 32mm，
        # 总开合约 64mm，可夹住 30mm 方块
        q0_mock = np.array([0.0, 1.17, 0.51, 0.0, 0.0, 0.0, np.deg2rad(220.0)])
        arm_interface = MockRealRobot(q0=q0_mock, num_joints=7, has_gripper=True)
    else:
        arm_interface = create_real_arm(
            hw_yaml=args.hw_yaml,
            channel=args.can,
            fallback_to_mock=True,
            hold=not args.no_hold,
        )
        if arm_interface is None:
            print("[mock mode] Using mock data because real hardware is unavailable.")
            arm_interface = MockRealRobot(q0=np.zeros(7), num_joints=7, has_gripper=True)
        else:
            print(f"Real robot connected via {args.can}.")

    if not getattr(arm_interface, "has_gripper", True):
        print("[WARN] Real robot configuration does not include a gripper; gripper feedback disabled.")
        return

    bridge = RealToSimBridge(
        robot,
        arm_interface=arm_interface,
        gripper_scale=args.gripper_scale,
        gripper_offset=args.gripper_offset,
    )
    grasp_feedback = GraspFeedback(robot, force_scale=args.force_scale)
    gravity_comp = GravityCompensator()

    arm_group = arm_interface.arm_group
    gripper_group = arm_interface.gripper_group

    dt = 1.0 / max(args.rate, 1.0)
    run_viewer = not args.headless

    # 首次读取真实状态并同步到 MuJoCo
    q_arm = arm_group.get_positions()
    q_gripper = gripper_group.get_positions()
    q_real = np.concatenate([q_arm, q_gripper])
    bridge.sync(q_real)

    print("Initial state synced.")
    print("  arm q (deg):", np.degrees(q_arm).round(2))
    print("  gripper q (deg):", np.degrees(q_gripper).round(2))
    print("Controls:")
    print(f"  arm kp/kd={args.kp_arm}/{args.kd_arm}")
    print(f"  gravity_scale per joint: {args.gravity_scale.round(3).tolist()}")
    print(f"  gripper kp/kd={args.kp_gripper}/{args.kd_gripper}")
    print(f"  force_scale={args.force_scale}, tau limits: arm={args.tau_arm_limit}, gripper={args.tau_gripper_limit}")
    if not args.no_hold:
        print("  [HINT] If the arm feels locked, run with --no-hold to disable motor holding torque.")
    print("Close the viewer or press Ctrl+C to stop.")

    def step_loop() -> tuple[float, float, dict]:
        """单次控制循环：读取 -> 同步 -> 臂重力补偿 -> 仿真步 -> 夹爪力反馈。"""
        # 1. 读取真实状态
        q_arm = arm_group.get_positions()
        dq_arm = arm_group.get_velocities()
        q_gripper = gripper_group.get_positions()
        dq_gripper = gripper_group.get_velocities()

        # 2. 同步到 MuJoCo
        q_real = np.concatenate([q_arm, q_gripper])
        bridge.sync(q_real)

        # 3. 计算并发送臂重力补偿命令
        tau_g = args.gravity_scale * gravity_comp.compute(q_arm)
        tau_arm = tau_g - args.kd_arm * dq_arm
        tau_arm = _limit_array(tau_arm, args.tau_arm_limit)
        arm_group.send_mit(
            pos=q_arm,
            vel=np.zeros(arm_group.num_joints),
            kp=np.full(arm_group.num_joints, args.kp_arm),
            kd=np.full(arm_group.num_joints, args.kd_arm),
            tau=tau_arm,
        )

        # 4. 推进 MuJoCo 仿真一步
        robot.step()

        # 5. 检测夹爪与方块接触，计算反馈力矩并发送给真实夹爪
        tau_feedback = grasp_feedback.compute_gripper_feedback_torque(data=robot.data)
        tau_feedback = float(np.clip(tau_feedback, -args.tau_gripper_limit, args.tau_gripper_limit))
        tau_gripper = tau_feedback + args.kp_gripper * (q_gripper - q_gripper) - args.kd_gripper * dq_gripper
        tau_gripper = _limit_array(tau_gripper, args.tau_gripper_limit)
        gripper_group.send_mit(
            pos=q_gripper,
            vel=np.zeros(gripper_group.num_joints),
            kp=np.full(gripper_group.num_joints, args.kp_gripper),
            kd=np.full(gripper_group.num_joints, args.kd_gripper),
            tau=tau_gripper,
        )

        return float(tau_feedback), float(np.linalg.norm(tau_g)), grasp_feedback.get_contact_info(robot.data)

    if run_viewer:
        print("Opening MuJoCo viewer...")
        with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
            try:
                while viewer.is_running():
                    t_start = time.perf_counter()
                    tau_feedback, tau_g_norm, info = step_loop()
                    viewer.sync()
                    print(
                        f"  tau_g={tau_g_norm:6.3f}N·m  "
                        f"F_L={info['left_force_N']:6.2f}N  F_R={info['right_force_N']:6.2f}N  "
                        f"tau_fb={tau_feedback:6.3f}N·m  grasp={info['is_grasping']}",
                        end="\r",
                        flush=True,
                    )
                    elapsed = time.perf_counter() - t_start
                    time.sleep(max(0.0, dt - elapsed))
            except KeyboardInterrupt:
                print("\nCtrl+C received, disabling robot...")
        print("\nViewer closed.")
    else:
        print("Running headless. Press Ctrl+C to stop.")
        try:
            while True:
                t_start = time.perf_counter()
                tau_feedback, tau_g_norm, info = step_loop()
                print(
                    f"  tau_g={tau_g_norm:6.3f}N·m  "
                    f"F_L={info['left_force_N']:6.2f}N  F_R={info['right_force_N']:6.2f}N  "
                    f"tau_fb={tau_feedback:6.3f}N·m  grasp={info['is_grasping']}",
                    end="\r",
                    flush=True,
                )
                elapsed = time.perf_counter() - t_start
                time.sleep(max(0.0, dt - elapsed))
        except KeyboardInterrupt:
            print("\nCtrl+C received, disabling robot...")

    bridge.close()
    print("[08_grasp_feedback] Exited safely.")


if __name__ == "__main__":
    main()
