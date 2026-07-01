# AGENTS.md

## 项目背景

本工程为 reBot Arm B601-RS 机械臂的 **MuJoCo 仿真** 项目。
底层运动学、逆运动学、动力学、重力补偿由第三方仓库
`reBotArm_control_py` 提供；本仓库负责 MuJoCo 模型、仿真循环、
控制接口以及 Real-to-Sim / Sim-to-Real 桥接。

## 重要约定

### 第三方 SDK 管理

- SDK 路径：`third_party/reBotArm_control_py`
- 通过 `bash scripts/setup_third_party.sh` 拉取/更新。
- `third_party/` 已加入 `.gitignore`，**禁止**将 SDK 源码提交到本仓库。
- `src/rebot_b601_rs_sim/config.py` 会自动将 SDK 目录加入 `sys.path`，
  工程内可直接 `import reBotArm_control_py`。

### 模型与坐标系

- B601-RS URDF：`third_party/reBotArm_control_py/urdf/00-arm-rs_asm-v3/urdf/00-arm-rs_asm-v3.urdf`
- MuJoCo 机器人模型：`assets/robot/rebot.xml`（由 `scripts/convert_urdf_to_mjcf.py` 生成）
- MuJoCo 场景：`assets/robot/scene.xml`（包含 `rebot.xml`、地面、灯光、坐标轴）
- 受控关节名称：`joint1` ~ `joint6`（见 `src/rebot_b601_rs_sim/config.py`）
- 末端执行器帧：`gripper_end`
- 关节单位：弧度（rad）

### 代码组织

- `src/rebot_b601_rs_sim/robot/`：MuJoCo 模型封装与状态读取
- `src/rebot_b601_rs_sim/control/`：IK、重力补偿、控制器
- `src/rebot_b601_rs_sim/bridge/`：Real-to-Sim / Sim-to-Real
- `src/rebot_b601_rs_sim/simulation/`：仿真主循环
- `examples/`：可独立运行的示例脚本
- `tests/`：pytest 单元测试

### MuJoCo 模型生成

- `assets/robot/rebot.xml` 由 `rebot_b601_rs_sim.utils.mujoco_utils.generate_rebot_xml()`
  自动从 SDK 的 B601-RS URDF 生成（通过 MuJoCo 的 `mj_saveLastXML`）。
- `assets/robot/scene.xml` 包含 `rebot.xml` 以及场景元素（地面、灯光、坐标轴）。
- `rebot.xml` 已加入 `.gitignore`，不应提交到仓库。
- 不要直接手写修改 `rebot.xml`；如需调整，应修改 URDF 或 `scene.xml`，
  然后删除 `rebot.xml` 让它在下次加载时自动重新生成。
- `scripts/convert_urdf_to_mjcf.py` 仅用于生成备用的简化 capsule 模型
  `rebot_simplified.xml`，主流程不使用。

### 命名风格

- Python 代码遵循 PEP 8，类型提示使用 `from __future__ import annotations`。
- MuJoCo 执行器命名约定：`actuator_jointN`。

### 测试

- 运行测试：`pytest tests/`
- 新增模块建议补充基础导入/接口测试。

### 提交前检查

- 确保 `third_party/` 下没有新增被跟踪文件。
- 确保 `assets/robot/b601_rs.xml` 与脚本生成结果一致。
