"""MuJoCo 场景加载工具 — 从 URDF 生成 rebot.xml 并加载 scene.xml。"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Tuple

import mujoco

from ..config import RS_URDF_PATH, SCENE_PATH


def _fix_urdf_mesh_paths(urdf_text: str, mesh_dir: Path) -> str:
    """将 URDF 中的相对 mesh 路径替换为绝对路径。"""
    mesh_dir_str = str(mesh_dir).replace(os.sep, "/")
    return re.sub(
        r'filename="meshes/([^"]+)"',
        lambda m: f'filename="{mesh_dir_str}/{m.group(1)}"',
        urdf_text,
    )


def generate_rebot_xml(output_path: Path | None = None) -> Path:
    """从 B601-RS URDF 生成 MuJoCo MJCF 格式的 rebot.xml。

    Args:
        output_path: 输出文件路径。默认使用 ``assets/robot/rebot.xml``。

    Returns:
        生成的 rebot.xml 路径。
    """
    rebot_path = Path(output_path) if output_path is not None else RS_URDF_PATH.parents[1] / "rebot.xml"
    urdf_path = RS_URDF_PATH
    pkg_dir = urdf_path.parents[1]  # urdf/00-arm-rs_asm-v3/
    mesh_dir = pkg_dir / "meshes"

    urdf_text = urdf_path.read_text(encoding="utf-8")
    urdf_text = _fix_urdf_mesh_paths(urdf_text, mesh_dir)

    with tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False) as tmp:
        tmp.write(urdf_text)
        tmp_urdf_path = tmp.name

    try:
        robot_model = mujoco.MjModel.from_xml_path(tmp_urdf_path)
        mujoco.mj_saveLastXML(str(rebot_path), robot_model)
    finally:
        os.unlink(tmp_urdf_path)

    return rebot_path


def load_mujoco_model() -> Tuple[mujoco.MjModel, mujoco.MjData]:
    """加载 MuJoCo 场景模型。

    流程:
        1. 读取 B601-RS URDF，把 mesh 路径转为绝对路径
        2. 用 MuJoCo 加载 URDF，再通过 mj_saveLastXML 保存为 rebot.xml
        3. 加载 scene.xml（scene.xml 会 include rebot.xml）

    scene.xml 里可以添加世界坐标轴、地板、目标点等环境元素；
    rebot.xml 由本函数自动生成，包含机械臂模型。

    Returns:
        (model, data)
    """
    scene_dir = Path(SCENE_PATH).parent
    rebot_path = scene_dir / "rebot.xml"

    # 自动生成 rebot.xml
    generate_rebot_xml(rebot_path)

    # 加载完整场景（scene.xml 中 include 了 rebot.xml）
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    return model, data


def load_scene() -> Tuple[mujoco.MjModel, mujoco.MjData]:
    """``load_mujoco_model`` 的别名，语义更明确。"""
    return load_mujoco_model()
