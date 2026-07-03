"""夹爪抓取反馈：检测 MuJoCo 中夹爪与方块的接触，并将接触力映射为真实夹爪电机的反馈力矩。"""

from __future__ import annotations

import numpy as np

from ..robot.model import RobotModel


class GraspFeedback:
    """检测 MuJoCo 中夹爪与方块的接触并计算反馈力矩。

    工作流程：
      1. 通过 body 名称找到左右夹爪手指和方块对应的 geom 索引。
      2. 每帧扫描 ``data.contact``，找出夹爪手指与方块之间的接触。
      3. 调用 ``mj_contactForce`` 读取接触力，并投影到夹爪滑动方向。
      4. 将接触力大小转换为需要叠加到真实 gripper 电机 MIT 命令中的力矩。
    """

    def __init__(
        self,
        robot: RobotModel,
        cube_body_name: str = "cube",
        gripper_body_names: tuple[str, str] = ("gripper_left", "gripper_right"),
        force_scale: float = 0.02,
    ) -> None:
        """
        Args:
            robot: MuJoCo 机器人模型封装。
            cube_body_name: 方块 body 名称。
            gripper_body_names: 左右夹爪手指 body 名称（顺序：[left, right]）。
            force_scale: 接触力（N）到反馈力矩（N·m）的缩放系数，需根据实际电机和夹爪刚度调试。
        """
        import mujoco

        self.robot = robot
        self.cube_body_name = cube_body_name
        self.gripper_body_names = list(gripper_body_names)
        self.force_scale = force_scale

        model = robot.model
        self.cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cube_body_name)
        if self.cube_body_id < 0:
            raise ValueError(f"Cube body '{cube_body_name}' not found in MuJoCo model")

        self.gripper_body_ids: list[int] = []
        for name in gripper_body_names:
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if gid < 0:
                raise ValueError(f"Gripper body '{name}' not found in MuJoCo model")
            self.gripper_body_ids.append(gid)

        # 预计算每个 body 拥有的 geom 索引集合
        self.cube_geoms = self._body_geom_set(self.cube_body_id)
        self.gripper_geom_sets = [self._body_geom_set(gid) for gid in self.gripper_body_ids]

        # 预计算每个夹爪手指的滑动轴（world 坐标系）
        self.gripper_slide_axes = [self._get_slide_axis(model, name) for name in gripper_body_names]

    def _body_geom_set(self, body_id: int) -> set[int]:
        """获取指定 body 拥有的所有 geom 索引。"""
        model = self.robot.model
        adr = model.body_geomadr[body_id]
        num = model.body_geomnum[body_id]
        return set(range(adr, adr + num))

    def _get_slide_axis(self, model, body_name: str) -> np.ndarray:
        """获取指定夹爪手指 body 的滑动轴在世界坐标系中的方向。

        这里通过该 body 下第一个滑动关节的 axis 并转换到世界坐标系得到。
        """
        import mujoco

        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        jnt_adr = model.body_jntadr[body_id]
        if jnt_adr < 0:
            return np.array([0.0, 1.0, 0.0])

        # 关节局部轴
        local_axis = model.jnt_axis[jnt_adr].copy()
        # 转换到世界坐标系：使用 body 的 xmat
        xmat = self.robot.data.xmat[body_id].reshape(3, 3)
        world_axis = xmat @ local_axis
        # 归一化
        norm = np.linalg.norm(world_axis)
        if norm > 1e-6:
            world_axis /= norm
        return world_axis

    def compute_contact_forces(self, data) -> np.ndarray:
        """计算左右夹爪手指与方块之间的接触力大小。

        必须在 ``mujoco.mj_step`` 之后调用，此时 ``data.contact`` 和 ``efc_force``
        已经更新。

        Returns:
            forces (2,)，单位 N。forces[0] 为左手指，forces[1] 为右手指。
        """
        import mujoco

        forces = np.zeros(len(self.gripper_body_ids))
        if data.ncon == 0:
            return forces

        model = self.robot.model

        # 重新计算滑动轴（因为 body 方向会随机械臂运动变化）
        self.gripper_slide_axes = [
            self._get_slide_axis(model, name) for name in self.gripper_body_names
        ]

        for finger_idx, finger_geoms in enumerate(self.gripper_geom_sets):
            total_force = 0.0
            for c_id in range(data.ncon):
                c = data.contact[c_id]
                g1, g2 = int(c.geom1), int(c.geom2)
                if not (
                    (g1 in self.cube_geoms and g2 in finger_geoms)
                    or (g2 in self.cube_geoms and g1 in finger_geoms)
                ):
                    continue

                # 读取接触力（6 维：法向 + 两个切向 + 扭转相关）
                force_buf = np.zeros(6)
                mujoco.mj_contactForce(model, data, c_id, force_buf)
                # 前三个分量是接触力的线性部分
                contact_force = force_buf[:3]
                force_mag = float(np.linalg.norm(contact_force))
                total_force += force_mag

            forces[finger_idx] = total_force

        return forces

    def compute_gripper_feedback_torque(
        self,
        forces: np.ndarray | None = None,
        data=None,
        scale: float | None = None,
    ) -> float:
        """将夹爪接触力映射为需要反馈到真实 gripper 电机的力矩。

        Args:
            forces: 左右手指接触力（N）。若为 None，则调用 ``compute_contact_forces`` 计算。
            data: MuJoCo data。forces 为 None 时必须提供。
            scale: 力矩缩放系数，覆盖构造时的 ``force_scale``。

        Returns:
            反馈力矩（N·m），叠加到 gripper.send_mit 的 ``tau`` 参数上。
        """
        if forces is None:
            if data is None:
                raise ValueError("Must provide either forces or data")
            forces = self.compute_contact_forces(data)
        forces = np.asarray(forces, dtype=float)
        s = scale if scale is not None else self.force_scale
        # 总接触力越大，电机需要输出越大力矩来维持/继续闭合
        return float(s * np.sum(forces))

    def is_grasping(self, data=None, force_threshold: float = 0.5) -> bool:
        """判断夹爪是否正在抓取方块（接触力超过阈值）。"""
        if data is None:
            raise ValueError("Must provide data")
        forces = self.compute_contact_forces(data)
        return bool(np.any(forces > force_threshold))

    def get_contact_info(self, data) -> dict:
        """获取详细的接触信息，便于调试。"""
        forces = self.compute_contact_forces(data)
        return {
            "left_force_N": float(forces[0]),
            "right_force_N": float(forces[1]),
            "total_force_N": float(np.sum(forces)),
            "is_grasping": self.is_grasping(data),
        }
