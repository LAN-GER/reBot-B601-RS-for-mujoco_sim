"""全局配置与路径管理。"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY_DIR = PROJECT_ROOT / "third_party"
SDK_DIR = THIRD_PARTY_DIR / "reBotArm_control_py"

# 自动将 SDK 加入 Python 搜索路径，使工程内可以直接 import reBotArm_control_py
if str(SDK_DIR) not in sys.path:
    sys.path.insert(0, str(SDK_DIR))

# B601-RS URDF 路径（来自 SDK）
RS_URDF_PATH = (
    SDK_DIR / "urdf" / "00-arm-rs_asm-v3" / "urdf" / "00-arm-rs_asm-v3.urdf"
)

# MuJoCo 模型路径（使用手动转换的 XML）
ROBOT_XML_PATH = PROJECT_ROOT / "assets" / "00_arm_rs_asm_v3" / "00_arm_rs_asm_v3.xml"
SCENE_PATH = PROJECT_ROOT / "assets" / "00_arm_rs_asm_v3" / "scene.xml"
MJCF_PATH = SCENE_PATH  # 默认加载带场景的完整模型

# 末端执行器帧名称（需与 SDK 配置一致）
END_EFFECTOR_FRAME = "gripper_end"

# 受控关节名称与数量（B601-RS 为 6 轴机械臂）
JOINT_NAMES = [f"joint{i}" for i in range(1, 7)]
NUM_JOINTS = len(JOINT_NAMES)


def ensure_sdk_available() -> None:
    """检查 SDK 是否已拉取，若未拉取则抛出提示。"""
    if not SDK_DIR.exists():
        raise FileNotFoundError(
            f"SDK not found at {SDK_DIR}. "
            "Please run: bash scripts/setup_third_party.sh"
        )
