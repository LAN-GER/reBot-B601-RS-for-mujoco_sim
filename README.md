# reBot-B601-RS-for-mujoco_sim

基于 **MuJoCo** 的 reBot Arm B601-RS 机械臂仿真工程。

底层运动学 / 逆运动学 / 动力学 / 重力补偿通过
[reBotArm_control_py](https://github.com/vectorBH6/reBotArm_control_py.git)
提供，本工程负责将其与 MuJoCo 仿真环境桥接，并提供 Real-to-Sim 接口。

## 功能规划

- [x] MuJoCo 仿真框架搭建
- [x] Conda 环境配置
- [x] 集成 `reBotArm_control_py` SDK
- [x] MuJoCo 中实现 IK（含交互式 Viewer 示例）
- [x] MuJoCo 中实现重力补偿（含交互式 Viewer 示例）
- [x] Real-to-Sim 接口（支持真实 B601-RS 硬件与模拟模式）

## 环境要求

- Ubuntu 22.04+（推荐）
- Python 3.10
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 或 Anaconda
- 支持 OpenGL 的显卡（运行 MuJoCo Viewer 时需要）

## 安装与使用

### 1. 克隆本仓库

```bash
git clone https://github.com/LAN-GER/reBot-B601-RS-for-mujoco_sim.git
cd reBot-B601-RS-for-mujoco_sim
```

### 2. 拉取底层 SDK

```bash
bash scripts/setup_third_party.sh
```

这会把 `reBotArm_control_py` 克隆到 `third_party/reBotArm_control_py`。
该目录已加入 `.gitignore`，不会提交到本仓库。

### 3. 创建并激活 Conda 环境

```bash
conda env create -f environment.yml
conda activate rebot-b601-rs-sim
```

环境名称为 `rebot-b601-rs-sim`，包含 MuJoCo 3.10、Pinocchio、NumPy、SciPy、PyYAML、pytest 等依赖。

> **注意**：如果你的系统安装了 ROS 2，ROS 的 `PYTHONPATH` 可能携带旧版 Pinocchio，导致版本冲突。
> `environment.yml` 已配置 `PYTHONPATH=""`，激活环境后会自动隔离。

### 4. 验证环境

```bash
python -c "import mujoco, pinocchio, numpy; print('MuJoCo:', mujoco.__version__, 'Pinocchio:', pinocchio.__version__)"
pytest tests/
```

### 5. 加载 MuJoCo 模型

本工程直接使用用户手动转换的 MuJoCo XML：

- 机器人模型：`assets/00_arm_rs_asm_v3/00_arm_rs_asm_v3.xml`
- 场景模型：`assets/00_arm_rs_asm_v3/scene.xml`（包含机器人、地面、灯光、世界坐标轴、台面、抓取方块）

本工程不再使用 `assets/robot/` 目录；所有 MuJoCo XML 均来自 `assets/00_arm_rs_asm_v3/`。

## 运行示例

### 基础示例（无界面批量运行）

```bash
# 加载模型并做简单物理仿真
python examples/01_load_model.py

# 将真实机器人状态同步到仿真（需先启动 CAN）
sudo ip link set can0 up type can bitrate 500000
python examples/04_real_to_sim.py

# 无硬件时使用模拟模式
python examples/04_real_to_sim.py --mock
```

### 带可视化窗口的基础示例

任意基础示例后加 `--viewer` 即可打开 MuJoCo Viewer：

```bash
python examples/01_load_model.py --viewer
python examples/04_real_to_sim.py --viewer
```

> **注意**：`04_real_to_sim.py` 默认会尝试连接真实机械臂；在无硬件环境中请使用 `--mock` 或 `--headless`。

### 交互式 MuJoCo Viewer 示例（需要图形界面）

```bash
# 终端输入目标位姿，机械臂在 MuJoCo 中实时运动
python examples/06_interactive_ik_mujoco.py

# 在 MuJoCo Viewer 中拖动关节，松手后关节悬浮在当前位置
python examples/07_interactive_gravity_compensation_mujoco.py
```

`06` 的交互命令：

- `x y z` 或 `x y z roll pitch yaw`：目标位姿（米 / 弧度）
- `b` / `home` / `zero`：回归零点
- `q` / `quit` / `exit`：退出

`07` 的交互命令：

- `b` / `home` / `zero`：回归零点
- `o` / `open`：张开夹爪
- `c` / `close`：闭合夹爪
- `q` / `quit` / `exit`：退出

### 真实机械臂 + MuJoCo 数字孪生（Real-to-Sim）

```bash
# 09：纯重力补偿 + 数字孪生同步（无夹爪力反馈，可自由开合夹爪）
python examples/09_real_to_sim_gravity_comp.py --no-hold

# 无硬件时使用模拟模式
python examples/09_real_to_sim_gravity_comp.py --mock --headless
```

### 运行测试

```bash
pytest tests/
```

## 常用命令速查

| 命令 | 说明 |
| --- | --- |
| `bash scripts/setup_third_party.sh` | 拉取/更新 SDK |
| `conda env create -f environment.yml` | 创建 Conda 环境 |
| `conda activate rebot-b601-rs-sim` | 激活环境 |
| `pytest tests/` | 运行单元测试 |
| `python examples/06_interactive_ik_mujoco.py` | 交互式 IK |
| `python examples/07_interactive_gravity_compensation_mujoco.py` | 交互式重力补偿 |
| `python examples/09_real_to_sim_gravity_comp.py --no-hold` | 真机重力补偿 + 数字孪生同步 |

## 项目结构

```
reBot-B601-RS-for-mujoco_sim/
├── README.md                          # 本文件
├── environment.yml                    # Conda 环境配置
├── pytest.ini                         # pytest 配置
├── scripts/
│   ├── setup_third_party.sh           # 拉取 SDK
│   └── convert_urdf_to_mjcf.py        # 备用简化模型生成脚本
├── third_party/
│   └── reBotArm_control_py/           # git clone 的 SDK（不提交）
├── assets/
│   └── 00_arm_rs_asm_v3/              # 用户手动转换的 MuJoCo 模型
│       ├── 00_arm_rs_asm_v3.xml       # 机器人模型
│       ├── scene.xml                  # 场景（含地面、灯光、坐标轴、台面、方块）
│       └── meshes/                    # STL 网格文件
├── src/rebot_b601_rs_sim/
│   ├── config.py                      # 路径与全局配置
│   ├── robot/                         # MuJoCo 模型/状态封装
│   ├── control/                       # IK、重力补偿、控制器
│   ├── bridge/                        # Real-to-Sim 桥接
│   ├── simulation/                    # 仿真主循环
│   └── utils/                         # 工具函数
├── examples/                          # 示例脚本
└── tests/                             # 单元测试
```

## 依赖说明

- Python 3.10
- MuJoCo >= 3.0
- Pinocchio（SDK 依赖）
- NumPy / SciPy / PyYAML
- pytest（测试）
- MeshCat（可选可视化）

## 注意事项

- `third_party/reBotArm_control_py` 由脚本自动拉取，不会提交到本仓库。
- `assets/00_arm_rs_asm_v3/` 下的 MuJoCo XML 由用户手动维护，是本仓库的主要模型文件。
- `04_real_to_sim.py` 已接入 `reBotArm_control_py` 的 `RebotArm`：连接真实机械臂前请确认 CAN 接口已启动；无硬件时会自动回退到模拟模式。
- `scripts/convert_urdf_to_mjcf.py` 仅作为备用的简化 capsule 模型生成脚本，主流程不使用。

## 常见问题

### Q: 运行示例时提示 `ModuleNotFoundError: No module named 'rebot_b601_rs_sim'`

确保在工程根目录运行，且 `pytest.ini` 已配置 `pythonpath = src`。
若直接使用 `python -c` 运行，需手动设置 `PYTHONPATH=src`：

```bash
PYTHONPATH=src python -c "import mujoco; from rebot_b601_rs_sim.config import SCENE_PATH; m = mujoco.MjModel.from_xml_path(str(SCENE_PATH)); print('loaded', m.nq, m.nv)"
```

### Q: 导入 Pinocchio 时报错 `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`

通常是系统 ROS 的 Pinocchio 被优先加载。`rebot-b601-rs-sim` 环境已设置 `PYTHONPATH=""` 隔离 ROS，
请确认已执行 `conda activate rebot-b601-rs-sim`。

### Q: MuJoCo Viewer 无法启动

MuJoCo Viewer 需要图形界面。在 SSH 或无显示器环境中，可运行无 viewer 的基础示例，
或使用 X11 转发 / VNC / 本地运行。
