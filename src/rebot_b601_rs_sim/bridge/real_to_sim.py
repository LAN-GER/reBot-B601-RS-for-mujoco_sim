"""Real-to-Sim：将真实机器人状态同步到 MuJoCo 仿真。"""

from __future__ import annotations

import subprocess
import time
from typing import TYPE_CHECKING

import numpy as np

from ..config import JOINT_NAMES, NUM_JOINTS
from ..robot.model import RobotModel

if TYPE_CHECKING:
    pass


class MockRealRobot:
    """无硬件时的模拟真实机器人，便于离线调试。"""

    is_mock = True

    def __init__(self, q0: np.ndarray | None = None, num_joints: int = NUM_JOINTS) -> None:
        self.num_joints = num_joints
        self.q = np.asarray(q0, dtype=float) if q0 is not None else np.zeros(num_joints)
        if self.q.shape[0] != num_joints:
            raise ValueError(f"Expected {num_joints} joint values, got {self.q.shape[0]}")

    @property
    def joint_names(self) -> list[str]:
        return list(JOINT_NAMES)

    def get_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.q.copy(), np.zeros(self.num_joints), np.zeros(self.num_joints)

    def disconnect(self) -> None:
        pass


class RebotArmClient:
    """封装 SDK ``RebotArm``，提供连接检查、状态读取与安全断开。"""

    def __init__(self, hw_yaml: str = "rebotarm_rs.yaml", channel: str = "can0") -> None:
        from reBotArm_control_py.actuator.rebotarm import RebotArm

        self.hw_yaml = hw_yaml
        self.channel = channel
        self.arm = RebotArm(hw_yaml=hw_yaml)
        self._connected = False

    def check_can(self) -> bool:
        """检查指定的 CAN 接口是否已启动。"""
        try:
            result = subprocess.run(
                ["ip", "link", "show", self.channel],
                capture_output=True,
                text=True,
                check=True,
            )
            return "state UP" in result.stdout or "UP" in result.stdout
        except Exception:
            return False

    def connect(self, enable: bool = True) -> None:
        """连接真实机械臂。

        连接前会检查 CAN 接口是否已启动；若未启动则抛出清晰提示。
        连接成功后默认对所有关节上使能，以便电机持续反馈状态。
        """
        if not self.check_can():
            raise RuntimeError(
                f"CAN interface '{self.channel}' is not up. "
                f"Run: sudo ip link set {self.channel} up type can bitrate 500000"
            )
        self.arm.connect()
        self._connected = True
        if enable:
            self.arm.enable_all()

    @property
    def joint_names(self) -> list[str]:
        return list(self.arm.joint_names)

    def get_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """读取真实机器人状态，返回 (q, dq, tau)。"""
        return self.arm.get_state()

    def disconnect(self) -> None:
        """安全断开：失能所有电机并标记为未连接。"""
        if self._connected:
            try:
                self.arm.disable_all()
            except Exception as exc:
                print(f"[RebotArmClient] disable_all failed: {exc}")
            self._connected = False

    def estop(self) -> None:
        """紧急停止：立即失能电机。"""
        self.disconnect()


class RealToSimBridge:
    """把真实机械臂的关节状态实时写入 MuJoCo，构建数字孪生。

    支持两种工作模式：
      1. 连接真实硬件：传入 ``RebotArmClient`` 实例。
      2. 模拟模式：不传入或传入 ``MockRealRobot``，使用模拟状态。

    关节映射按照名称进行，确保真实机器人与 MuJoCo 模型的关节顺序一致。
    """

    def __init__(
        self,
        robot: RobotModel,
        arm_interface: RebotArmClient | MockRealRobot | None = None,
    ) -> None:
        """
        Args:
            robot: MuJoCo 机器人模型封装。
            arm_interface: 真实机械臂客户端。若为 None，则使用模拟模式。
        """
        self.robot = robot
        self.arm_interface = arm_interface
        self._mock_mode = arm_interface is None or getattr(arm_interface, "is_mock", False)

        if arm_interface is None:
            self.arm_interface = MockRealRobot()

        real_joint_names = self.arm_interface.joint_names
        self._real_indices: list[int] = []
        missing: list[str] = []
        for name in JOINT_NAMES:
            try:
                self._real_indices.append(real_joint_names.index(name))
            except ValueError:
                missing.append(name)
        if missing:
            raise ValueError(
                f"Real robot joint list {real_joint_names} is missing "
                f"joints expected by simulation: {missing}"
            )

    @property
    def is_mock(self) -> bool:
        """是否为模拟模式（无真实硬件连接）。"""
        return self._mock_mode

    def read_real_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """读取真实机械臂完整状态。

        Returns:
            (q, dq, tau)，维度均为 (num_real_joints,)。
        """
        q_full, dq_full, tau_full = self.arm_interface.get_state()
        return (
            np.asarray(q_full, dtype=float),
            np.asarray(dq_full, dtype=float),
            np.asarray(tau_full, dtype=float),
        )

    def sync(self, q_real: np.ndarray | None = None) -> None:
        """将真实状态写入仿真。

        Args:
            q_real: 若提供，直接使用该 6-DOF 关节位置；否则从硬件读取。
        """
        import mujoco

        if q_real is not None:
            q_mapped = np.asarray(q_real, dtype=float)
            if q_mapped.shape[0] != NUM_JOINTS:
                raise ValueError(
                    f"Expected {NUM_JOINTS} joint values, got {q_mapped.shape[0]}"
                )
            dq = np.zeros(NUM_JOINTS)
        else:
            q_full, dq_full, _ = self.read_real_state()
            q_mapped = q_full[self._real_indices]
            dq = dq_full[self._real_indices]

        self.robot.set_q(q_mapped, forward=False)

        # 写入关节速度（MuJoCo 中 1-DoF 关节的 qvel 地址与 qpos 地址相同）
        for addr, v in zip(self.robot.joint_qpos_addrs, dq):
            if addr < self.robot.model.nv:
                self.robot.data.qvel[addr] = v

        mujoco.mj_forward(self.robot.model, self.robot.data)

    def close(self) -> None:
        """断开真实机械臂连接（模拟模式下无操作）。"""
        if self.arm_interface is not None and hasattr(self.arm_interface, "disconnect"):
            self.arm_interface.disconnect()

    def __enter__(self) -> "RealToSimBridge":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def create_real_arm(
    hw_yaml: str = "rebotarm_rs.yaml",
    channel: str = "can0",
    fallback_to_mock: bool = False,
) -> RebotArmClient | None:
    """尝试连接真实机械臂，失败时可选回退到模拟模式。

    Args:
        hw_yaml: SDK 硬件配置文件名。
        channel: CAN 接口名，默认 can0。
        fallback_to_mock: 连接失败时是否返回 None（让调用方使用 mock 模式）。

    Returns:
        已连接的 ``RebotArmClient``，或 ``None``（仅当 fallback_to_mock=True 且失败时）。

    Raises:
        RuntimeError: 真实硬件不可用且 fallback_to_mock=False 时。
    """
    try:
        client = RebotArmClient(hw_yaml=hw_yaml, channel=channel)
        client.connect()
        return client
    except Exception as exc:
        if fallback_to_mock:
            print(f"[WARN] Failed to connect to real robot: {exc}")
            print("[WARN] Falling back to mock mode.")
            return None
        raise
