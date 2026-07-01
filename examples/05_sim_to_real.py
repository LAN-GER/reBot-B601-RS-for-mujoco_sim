#!/usr/bin/env python3
"""示例 5：将 MuJoCo 仿真控制指令下发到真实机器人。

用法:
    python examples/05_sim_to_real.py
    python examples/05_sim_to_real.py --viewer   # 打开 MuJoCo 可视化窗口
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

from rebot_b601_rs_sim.bridge.sim_to_real import SimToRealBridge
from rebot_b601_rs_sim.control.ik import IKSolver
from rebot_b601_rs_sim.robot.model import RobotModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Send MuJoCo control commands to real robot.")
    parser.add_argument("--viewer", action="store_true", help="Open MuJoCo viewer window")
    args = parser.parse_args()

    robot = RobotModel()
    ik = IKSolver()
    bridge = SimToRealBridge(arm_interface=None)

    q_test = np.array([0.0, 0.5, 1.0, 0.0, 0.5, 0.0])
    target_pos, target_rot = ik.forward_kinematics(q_test)
    print(f"Target pose from FK: pos={target_pos}")

    q_target, success = ik.solve(target_pos, target_rot=target_rot, q_init=np.zeros(6))
    print(f"IK success: {success}, q_target: {q_target}")
    if not success:
        print("IK failed, exiting.")
        return

    # 仿真中直接设置关节位置（无 actuator 时采用运动学驱动）
    robot.reset(q_target)
    q_cmd = robot.get_q()
    print(f"Sending position command to real robot: {q_cmd}")

    if args.viewer:
        with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
            for _ in range(1000):
                if not viewer.is_running():
                    break
                viewer.sync()
                time.sleep(float(robot.model.opt.timestep))
        print("Viewer closed.")


if __name__ == "__main__":
    main()
