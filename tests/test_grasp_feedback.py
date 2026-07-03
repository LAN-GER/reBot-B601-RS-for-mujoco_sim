"""测试夹爪与方块的接触检测及力反馈计算。"""

from __future__ import annotations

import numpy as np

import mujoco

from rebot_b601_rs_sim.bridge.grasp_feedback import GraspFeedback
from rebot_b601_rs_sim.robot.model import RobotModel


def test_grasp_feedback_detects_contact() -> None:
    """当夹爪闭合在方块上时，应检测到非零接触力。"""
    robot = RobotModel()
    gf = GraspFeedback(robot)

    # 把机械臂放到方块正上方可抓取的位置（台面高度减半后）
    q_arm = np.array([0.0, 1.17, 0.51, 0.0, 0.0, 0.0])
    robot.set_q(q_arm, forward=True)

    # 将夹爪设置为较小的开合（闭合趋势），并让 MuJoCo 物理稳定
    # 真实 gripper 0~345deg 对应 MuJoCo 0~50mm；这里用 45deg -> 约 6.2mm 单侧
    gripper_rad = np.deg2rad(45.0)
    disp = 0.00830 * gripper_rad
    robot.data.qpos[6] = disp  # joint_left
    robot.data.qpos[7] = disp  # joint_right

    # 让方块落在台子上并稳定
    mujoco.mj_forward(robot.model, robot.data)
    for _ in range(10):
        mujoco.mj_step(robot.model, robot.data)

    forces = gf.compute_contact_forces(robot.data)
    print(f"contact forces: {forces}", flush=True)
    assert forces.shape == (2,)
    assert np.any(forces > 0.1), "Expected contact force when gripper is closed on cube"


def test_grasp_feedback_no_contact_when_open() -> None:
    """当夹爪张开时，不应检测到显著接触力。"""
    robot = RobotModel()
    gf = GraspFeedback(robot)

    q_arm = np.array([0.0, 1.17, 0.51, 0.0, 0.0, 0.0])
    robot.set_q(q_arm, forward=True)

    # 夹爪张开：30deg -> 约 4.3mm 单侧，应不会碰到方块
    gripper_rad = np.deg2rad(30.0)
    disp = 0.00830 * gripper_rad
    robot.data.qpos[6] = disp
    robot.data.qpos[7] = disp

    mujoco.mj_forward(robot.model, robot.data)
    for _ in range(200):
        mujoco.mj_step(robot.model, robot.data)

    forces = gf.compute_contact_forces(robot.data)
    print(f"open gripper forces: {forces}", flush=True)
    assert np.all(forces < 0.1), "Did not expect contact force with open gripper"


def test_grasp_feedback_torque_mapping() -> None:
    """接触力到反馈力矩的映射应符合 scale。"""
    robot = RobotModel()
    gf = GraspFeedback(robot, force_scale=0.05)

    tau = gf.compute_gripper_feedback_torque(forces=np.array([2.0, 3.0]))
    assert np.isclose(tau, 0.05 * 5.0)
