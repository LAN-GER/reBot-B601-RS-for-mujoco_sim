#!/usr/bin/env python3
"""示例 4：将真实机器人状态同步到 MuJoCo 仿真（数字孪生）。

用法:
    python examples/04_real_to_sim.py
    python examples/04_real_to_sim.py --viewer   # 打开 MuJoCo 可视化窗口
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rebot_b601_rs_sim.bridge.real_to_sim import RealToSimBridge
from rebot_b601_rs_sim.robot.model import RobotModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync real robot state to MuJoCo simulation.")
    parser.add_argument("--viewer", action="store_true", help="Open MuJoCo viewer window")
    args = parser.parse_args()

    robot = RobotModel()
    bridge = RealToSimBridge(robot)

    q_real = np.array([0.2, 0.5, 1.0, -0.3, 0.1, 0.0])
    dq_real = np.zeros(6)
    bridge.sync(q_real, dq_real)
    print(f"Synced real state to sim: q={robot.get_q()}, dq={robot.get_dq()}")

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
