"""Sim-to-Real：将 MuJoCo 计算的控制量下发到真实机器人。"""

from __future__ import annotations

import numpy as np


class SimToRealBridge:
    """把仿真中计算得到的控制指令转换为真实机器人可接收的格式。"""

    def __init__(self, arm_interface: object | None = None) -> None:
        """
        Args:
            arm_interface: SDK 中的真实机械臂接口，例如 ``reBotArm_control_py.actuator.rebotarm.RebotArm``。
                           仅在连接真实硬件时使用。
        """
        self.arm_interface = arm_interface

    def send_position_command(self, q_cmd: np.ndarray) -> None:
        """下发位置指令。

        TODO: 接入 SDK actuator 接口，例如调用 arm_interface.set_joint_positions(q_cmd)。
        """
        if self.arm_interface is None:
            return
        # 示例：self.arm_interface.set_joint_positions(q_cmd)
        raise NotImplementedError("Hardware interface not integrated yet.")

    def send_torque_command(self, tau_cmd: np.ndarray) -> None:
        """下发力矩指令。

        TODO: 接入 SDK actuator 接口。
        """
        if self.arm_interface is None:
            return
        raise NotImplementedError("Hardware interface not integrated yet.")

    def send_mit_command(
        self,
        q_cmd: np.ndarray,
        dq_cmd: np.ndarray,
        kp: float,
        kd: float,
        tau_ff: np.ndarray,
    ) -> None:
        """下发 MIT 控制模式指令（位置 + 速度 + 前馈力矩）。

        TODO: 接入 SDK actuator 接口。
        """
        if self.arm_interface is None:
            return
        raise NotImplementedError("Hardware interface not integrated yet.")
