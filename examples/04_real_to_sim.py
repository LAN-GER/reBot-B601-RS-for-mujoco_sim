#!/usr/bin/env python3
"""示例 4：将真实机器人状态实时同步到 MuJoCo 仿真（数字孪生）。

用法:
    # 连接真实机械臂（需先启动 CAN 接口）
    python examples/04_real_to_sim.py

    # 模拟模式（无硬件，使用测试数据）
    python examples/04_real_to_sim.py --mock

    # 仅无窗口运行，输出关节角度
    python examples/04_real_to_sim.py --headless

真实机械臂连接前请确保：
    sudo ip link set can0 up type can bitrate 500000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rebot_b601_rs_sim.bridge.real_to_sim import (
    MockRealRobot,
    RealToSimBridge,
    create_real_arm,
)
from rebot_b601_rs_sim.robot.model import RobotModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync real robot state to MuJoCo simulation.")
    parser.add_argument("--mock", action="store_true", help="Use mock data instead of real hardware")
    parser.add_argument("--headless", action="store_true", help="Run without MuJoCo viewer window")
    parser.add_argument("--can", type=str, default="can0", help="CAN interface name (default: can0)")
    parser.add_argument(
        "--hw-yaml",
        type=str,
        default="rebotarm_rs.yaml",
        help="SDK hardware configuration YAML (default: rebotarm_rs.yaml)",
    )
    args = parser.parse_args()

    print("[04_real_to_sim] Starting...", flush=True)

    robot = RobotModel()

    if args.mock:
        print("[mock mode] No real hardware connected, using synthetic joint state.")
        arm_interface = MockRealRobot()
    else:
        arm_interface = create_real_arm(
            hw_yaml=args.hw_yaml,
            channel=args.can,
            fallback_to_mock=True,
        )
        if arm_interface is None:
            print("[mock mode] Using mock data because real hardware is unavailable.")
            arm_interface = MockRealRobot()
        else:
            print(f"Real robot connected via {args.can}.")

    with RealToSimBridge(robot, arm_interface=arm_interface) as bridge:
        if bridge.is_mock:
            t0 = time.time()
            q_mock = np.zeros(6)
        else:
            q_real, _, _ = bridge.read_real_state()
            bridge.sync(q_real)
            print(f"Initial real state synced: q={np.degrees(q_real[:6]).round(2)} deg")

        dt = float(robot.model.opt.timestep)

        run_viewer = not args.headless

        def step_sync() -> None:
            nonlocal q_mock
            if bridge.is_mock:
                elapsed = time.time() - t0
                q_mock = 0.3 * np.sin(2.0 * elapsed + np.arange(6))
                bridge.sync(q_mock)
            else:
                bridge.sync()

        if run_viewer:
            print("Opening MuJoCo viewer. Close the window to stop.")
            with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
                try:
                    while viewer.is_running():
                        step_sync()
                        viewer.sync()
                        time.sleep(dt)
                except KeyboardInterrupt:
                    print("\nCtrl+C received, disabling robot...")
            print("Viewer closed.")
        else:
            print("Running headless real-to-sim sync. Press Ctrl+C to stop.")
            try:
                while True:
                    step_sync()
                    q_sim = robot.get_q()
                    print(f"  Sim q (deg): {np.degrees(q_sim).round(2)}", end="\r")
                    time.sleep(dt)
            except KeyboardInterrupt:
                print("\nCtrl+C received, disabling robot...")

    print("[04_real_to_sim] Exited safely.")


if __name__ == "__main__":
    main()
