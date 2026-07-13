#!/usr/bin/env python3
"""交互式 IK + MuJoCo 物理仿真可视化。

- 臂关节 1~6 使用 <motor> 作动器，由 Python POS_VEL 串级控制器驱动：
      vel_cmd = clip(pos_kp * (q_target - q), -vlim, vlim)
      torque  = clip(vel_kp * (vel_cmd - qd) + vel_ki * integral, -tau_max, tau_max)

- 夹爪驱动关节 7 使用 <motor> 作动器，由 Python PD 控制器驱动：
      torque = kp_gripper * (q_target - q) - kv_gripper * qd

joint_left / joint_right 通过 XML 中的 equality 约束与 joint7 联动，
不需要额外作动器。

不额外添加重力前馈；机械臂/夹爪的稳定完全依赖控制器。

用法:
    python examples/06_interactive_ik_mujoco.py

交互:
    输入: x y z [roll pitch yaw]  (米 / 弧度)
    例: 0.3 0 0.2
    例: 0.3 0 0.2 0 0 0
    b / home / zero: 回归零点
    o / open: 张开夹爪
    c / close: 闭合夹爪
    q / quit / exit: 退出
"""

from __future__ import annotations

import queue
import signal
import sys
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import pinocchio as pin

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rebot_b601_rs_sim.config import SCENE_PATH
from rebot_b601_rs_sim.control.ik import IKSolver
from rebot_b601_rs_sim.control.pos_vel_controller import POSVELController

# 机械臂关节数（不含 gripper 驱动关节 joint7）
N_ARM_JOINTS = 6
LINEAR_SPEED = 0.15  # 笛卡尔运动速度 (m/s)，用于估算轨迹时长

# 夹爪：真实 7 号电机 0°（闭合）~ 345°（张开）→ MuJoCo joint7 直线位移 0 ~ 0.05 m
GRIPPER_DEG_MAX = 345.0
GRIPPER_DISP_MAX = 0.05

# 臂关节真实电机 POS_VEL 模式参数（与 rebotarm_rs.yaml 对应）
POS_KP = np.array([13.0, 16.0, 14.0, 20.0, 10.0, 10.0])
VEL_KP = np.array([12.0, 14.0, 14.0, 5.0, 4.0, 4.0])
VEL_KI = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
VLIM = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
TAU_MAX = np.array([36.0, 36.0, 36.0, 36.0, 36.0, 36.0])

# 力矩输出低通滤波系数（越小越平滑，1.0 表示无滤波）
ARM_OUTPUT_FILTER_ALPHA = 0.3

# 夹爪 PD 控制器参数（在 Python 中可调）
# 增大 kp/kv 可提高夹爪刚度和响应速度，减少臂运动时的被动开合。
# 但过大容易引起振荡；700/70/300 是响应速度与稳定性的折中。
# BIAS 用于接近目标时提供额外力矩，克服静摩擦，避免最后一点闭合过慢。
GRIPPER_KP = 700.0
GRIPPER_KV = 70.0
GRIPPER_TAU_MAX = 300.0
GRIPPER_BIAS = 2.0


def input_thread_fn(cmd_queue: queue.Queue, stop_event: threading.Event) -> None:
    """后台线程：读取终端输入并放入队列。"""
    while not stop_event.is_set():
        try:
            line = input("目标位姿 > ")
        except EOFError:
            cmd_queue.put("quit")
            break
        if line.strip():
            cmd_queue.put(line.strip())


def min_jerk_interpolation(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Min-jerk 归一化插值，t in [0, 1]。"""
    t = float(np.clip(t, 0.0, 1.0))
    s = 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5
    return q0 + (q1 - q0) * s


def main() -> None:
    # ── 加载模型 ──────────────────────────────────────────────────────────────
    mj_model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    mj_data = mujoco.MjData(mj_model)

    ik = IKSolver()

    # 仿真时间步
    dt = float(mj_model.opt.timestep)
    if dt <= 0.0:
        dt = 0.002
        mj_model.opt.timestep = dt

    # 查找臂关节 <motor> 作动器索引
    arm_actuator_names = [f"joint{i}_motor" for i in range(1, N_ARM_JOINTS + 1)]
    arm_actuator_ids: list[int] = []
    for name in arm_actuator_names:
        aid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            raise RuntimeError(f"MuJoCo 模型中未找到作动器: {name}")
        arm_actuator_ids.append(aid)

    # 查找夹爪驱动关节 <motor> 作动器索引
    gripper_actuator_name = "joint7_motor"
    gripper_actuator_id = mujoco.mj_name2id(
        mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, gripper_actuator_name
    )
    if gripper_actuator_id < 0:
        raise RuntimeError(f"MuJoCo 模型中未找到作动器: {gripper_actuator_name}")

    # 查找夹爪驱动关节 joint7 在 qpos 中的地址
    def _joint_qpos_addr(name: str) -> int:
        jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"MuJoCo 模型中未找到关节: {name}")
        return int(mj_model.jnt_qposadr[jid])

    gripper7_addr = _joint_qpos_addr("joint7")

    # POS_VEL 串级控制器（仅用于臂关节）
    pos_vel = POSVELController(
        pos_kp=POS_KP,
        vel_kp=VEL_KP,
        vel_ki=VEL_KI,
        vlim=VLIM,
        tau_max=TAU_MAX,
        dt=dt,
        output_filter_alpha=ARM_OUTPUT_FILTER_ALPHA,
    )

    # 当前关节配置：用模型默认 qpos 初始化，保留 cube 等未控制自由度的初始位姿
    q_current = mj_data.qpos.copy()

    # 夹爪目标角度（度），0° 闭合，345° 张开
    gripper_target_deg = 0.0

    # 设置 MuJoCo 初始状态
    mujoco.mj_forward(mj_model, mj_data)

    # ── 交互线程 ──────────────────────────────────────────────────────────────
    cmd_queue: queue.Queue[str] = queue.Queue()
    stop_event = threading.Event()

    def _signal_handler(sig, frame) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, _signal_handler)

    input_thread = threading.Thread(
        target=input_thread_fn,
        args=(cmd_queue, stop_event),
        daemon=True,
    )
    input_thread.start()

    # 当前轨迹状态 (q_start, q_end, t_start, duration)
    trajectory: tuple[np.ndarray, np.ndarray, float, float] | None = None
    trajectory_lock = threading.Lock()

    print("=" * 60)
    print("MuJoCo IK 物理仿真已启动")
    print("臂作动器: <motor> + Python POS_VEL 控制器")
    print("夹爪作动器: <motor> + Python PD 控制器")
    print("输入: x y z [roll pitch yaw] (米 / 弧度)")
    print("      b / home / zero: 回归零点")
    print("      o / open: 张开夹爪")
    print("      c / close: 闭合夹爪")
    print("      q / quit / exit: 退出")
    print("=" * 60)

    def start_trajectory(q_end: np.ndarray, duration: float | None = None) -> None:
        """从当前实际关节角出发，规划一条到 q_end 的关节空间轨迹。"""
        nonlocal trajectory
        q_actual = mj_data.qpos[:N_ARM_JOINTS].copy()
        if duration is None:
            dist = float(np.linalg.norm(q_end - q_actual))
            duration = max(1.0, dist / 0.5)  # 粗略按 0.5 rad/s 估算
        with trajectory_lock:
            trajectory = (
                q_actual.copy(),
                q_end.copy(),
                time.time(),
                duration,
            )

    def print_current_pose() -> None:
        pos, rot = ik.forward_kinematics(mj_data.qpos[:N_ARM_JOINTS])
        rpy = pin.rpy.matrixToRpy(rot)
        print(
            f"  当前末端: pos=[{pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}] "
            f"rpy=[{rpy[0]:.3f} {rpy[1]:.3f} {rpy[2]:.3f}]"
        )

    print_current_pose()

    # ── 主仿真循环 ────────────────────────────────────────────────────────────
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running() and not stop_event.is_set():
            # 处理新的终端命令
            try:
                line = cmd_queue.get_nowait()
            except queue.Empty:
                line = None

            if line is not None:
                cmd = line.lower()
                if cmd in ("q", "quit", "exit"):
                    stop_event.set()
                    break

                if cmd in ("b", "home", "zero"):
                    print("  回归零点")
                    start_trajectory(np.zeros(N_ARM_JOINTS))
                    continue

                if cmd in ("o", "open"):
                    print("  张开夹爪")
                    gripper_target_deg = GRIPPER_DEG_MAX
                    continue

                if cmd in ("c", "close"):
                    print("  闭合夹爪")
                    gripper_target_deg = 0.0
                    continue

                try:
                    vals = [float(x) for x in line.split()]
                    if len(vals) not in (3, 6):
                        print("  需要 3 个值（仅位置）或 6 个值（位置+姿态）")
                        continue
                except ValueError:
                    print("  无效输入，格式: x y z [roll pitch yaw]")
                    continue

                x, y, z = vals[0], vals[1], vals[2]
                roll = vals[3] if len(vals) >= 6 else 0.0
                pitch = vals[4] if len(vals) >= 6 else 0.0
                yaw = vals[5] if len(vals) >= 6 else 0.0

                target_pos = np.array([x, y, z])
                target_rot = (
                    pin.rpy.rpyToMatrix(roll, pitch, yaw)
                    if len(vals) == 6
                    else None
                )

                print(
                    f"  目标: pos=[{x:.3f} {y:.3f} {z:.3f}] "
                    f"rpy=[{roll:.3f} {pitch:.3f} {yaw:.3f}]"
                )

                # 用 MuJoCo 当前实际关节角作为 IK 初始猜测
                q_actual = mj_data.qpos[:N_ARM_JOINTS].copy()
                q_target, success = ik.solve(
                    target_pos, target_rot=target_rot, q_init=q_actual
                )
                if not success:
                    print("  IK 未收敛")
                    continue

                print(
                    f"  IK 成功: 关节(deg)="
                    f"{np.degrees(q_target).round(2).tolist()}"
                )

                # 根据目标距离估算轨迹时长
                cur_pos, _ = ik.forward_kinematics(q_actual)
                duration = max(
                    1.0,
                    float(np.linalg.norm(target_pos - cur_pos)) / LINEAR_SPEED,
                )
                start_trajectory(q_target, duration)

            # 更新关节空间轨迹
            with trajectory_lock:
                if trajectory is not None:
                    q_start, q_end, t_start, duration = trajectory
                    elapsed = time.time() - t_start
                    if elapsed >= duration:
                        q_current[:N_ARM_JOINTS] = q_end
                        trajectory = None
                        print("  到达目标")
                        print_current_pose()
                    else:
                        q_current[:N_ARM_JOINTS] = min_jerk_interpolation(
                            q_start, q_end, elapsed / duration
                        )

            # 夹爪目标位置（m）：电机角度 -> joint7 直线位移
            gripper_disp = (gripper_target_deg / GRIPPER_DEG_MAX) * GRIPPER_DISP_MAX
            gripper_disp = float(np.clip(gripper_disp, 0.0, GRIPPER_DISP_MAX))

            # 夹爪 PD 控制器（带摩擦补偿偏置）
            q7 = mj_data.qpos[gripper7_addr]
            qd7 = mj_data.qvel[gripper7_addr]
            gripper_err = gripper_disp - q7
            tau_gripper = (
                GRIPPER_KP * gripper_err
                - GRIPPER_KV * qd7
                + GRIPPER_BIAS * np.sign(gripper_err)
            )
            tau_gripper = float(np.clip(tau_gripper, -GRIPPER_TAU_MAX, GRIPPER_TAU_MAX))
            mj_data.ctrl[gripper_actuator_id] = tau_gripper

            # 臂关节 POS_VEL 控制器
            q_target = q_current[:N_ARM_JOINTS]
            q = mj_data.qpos[:N_ARM_JOINTS]
            qd = mj_data.qvel[:N_ARM_JOINTS]
            tau = pos_vel.compute(q_target, q, qd)

            # 将力矩写入臂 <motor> 作动器
            for i, aid in enumerate(arm_actuator_ids):
                mj_data.ctrl[aid] = float(tau[i])

            # 物理积分一步
            mujoco.mj_step(mj_model, mj_data)
            viewer.sync()

    print("\n退出仿真。")


if __name__ == "__main__":
    main()
