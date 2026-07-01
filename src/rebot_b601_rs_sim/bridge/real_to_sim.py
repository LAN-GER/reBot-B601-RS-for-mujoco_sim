"""Real-to-Sim：将真实机器人状态同步到 MuJoCo 仿真。"""

from __future__ import annotations

import numpy as np

from ..robot.model import RobotModel


class RealToSimBridge:
    """把真实机械臂的关节状态写入 MuJoCo，构建数字孪生。"""

    def __init__(self, robot: RobotModel) -> None:
        self.robot = robot

    def sync(
        self,
        q_real: np.ndarray,
        dq_real: np.ndarray | None = None,
        tau_real: np.ndarray | None = None,
    ) -> None:
        """将真实状态写入仿真。

        Args:
            q_real: 真实关节位置 (NUM_JOINTS,)。
            dq_real: 真实关节速度 (NUM_JOINTS,)，可选。
            tau_real: 真实关节力矩 (NUM_JOINTS,)，可选（可写入 sensor 或 qfrc）。
        """
        self.robot.set_q(q_real, forward=False)

        if dq_real is not None:
            dq_real = np.asarray(dq_real, dtype=float)
            for i, addr in enumerate(self.robot.joint_qpos_addrs):
                # MuJoCo 中 qvel 的地址通常与 qpos 地址相同（对于 1-DoF 关节）
                if addr < self.robot.model.nv:
                    self.robot.data.qvel[addr] = dq_real[i]

        # TODO: 若需显示真实力矩，可写入 qfrc_applied 或自定义 sensor
        _ = tau_real

        # 前向传播使运动学一致
        import mujoco

        mujoco.mj_forward(self.robot.model, self.robot.data)
