# MPC-UAV 项目详解

这个项目实现了一个基于高斯过程 (GP) 增强的模型预测控制 (MPC) 系统，专用于四旋翼无人机的高速轨迹跟踪。核心思想是利用 MPC 处理系统的基本动力学，并利用 GP 在线或离线学习动力学模型的残差（即模型无法预测的误差，如空气阻力、各种扰动等），从而提高控制精度。

## 1. 目录结构概览

项目遵循典型的 Python 科学计算项目结构，核心逻辑位于 `src/`，配置位于 `config/`，数据和输出分别在 `data/` 和 `outputs/`。

```
mpc_uav/
├── src/                    # 源代码目录
│   ├── core/               # [核心] MPC 控制与动力学模型
│   ├── gp/                 # [核心] 高斯过程 (GP) 学习模块
│   ├── experiments/        # 实验与测试脚本 (入口点)
│   ├── visualization/      # 可视化工具
│   └── utils/              # 通用工具 (数学计算、轨迹生成等)
├── config/                 # 配置文件 (系统参数、GP参数)
├── acados_models/          # ACADOS 自动生成的 C 代码 (MPC求解器)
├── outputs/                # 仿真结果、图片、训练好的模型
├── data/                   # 原始实验数据
└── README.md               # 项目说明
```

## 2. 核心模块详解 (src)

### 2.1 Core (`src/core/`) - 控制大脑
这是项目的核心部分，负责定义物理模型和构建 MPC 控制器。

*   **`dynamics.py` (`Quadrotor3D`)**:
    *   定义四旋翼的物理动力学模型 (First Principles Model)。
    *   包含运动方程 $\dot{x} = f(x, u)$。
    *   提供 `step` 或 `update` 方法用于仿真环境中的状态推进。

*   **`optimizer.py` (`Quad3DOptimizer`)**:
    *   这是与 **ACADOS** 库交互的接口。
    *   定义了最优控制问题 (OCP)：目标函数 (Cost Function)、约束条件 (Constraints)。
    *   将 GP 预测的动力学残差项嵌入到 MPC 的预测模型中：$x_{k+1} = f_{nominal}(x_k, u_k) + GP(x_k, u_k)$。

*   **`controller.py` (`Quad3DMPC`)**:
    *   控制器的顶层类，封装了 `Quad3DOptimizer`。
    *   `optimize()`: 接收当前状态，调用 ACADOS 求解器计算最优控制律。
    *   `simulate()`: 调用 `dynamics.py` 进行仿真步进（用于纯仿真模式）。
    *   管理 GP 模型的加载和预测。

### 2.2 GP (`src/gp/`) - 学习引擎
负责学习名义模型 (Nominal Model) 与真实世界之间的误差。

*   **`base.py`**: 定义了基本的 GP 回归器类和核函数 (Kernel)。
*   **`offline.py`**: 离线训练脚本。从 `data/` 读取飞行日志，训练 GP 模型，并将模型保存为 `.pkl` 文件。
*   **`online.py`**: 在线 GP 实现。支持在飞行过程中实时更新 GP 模型（增量学习），适应动态环境。
*   **`utils.py`**: 数据处理工具，包括 `GPDataset` 类，用于加载、预处理和聚类训练数据。

### 2.3 Experiments (`src/experiments/`) - 实验入口
包含各种场景的测试脚本，是用户运行代码的主要入口。

*   **`trajectory_test.py`**:
    *   **主仿真脚本**。
    *   加载配置和模型。
    *   生成参考轨迹 (Circle, Lemniscate)。
    *   运行控制循环：`测量状态 -> MPC 优化 -> 执行控制 -> 物理仿真`。
    *   调用可视化模块显示结果。

*   **`point_tracking_and_record.py`**:
    *   简单定点飞行脚本。
    *   主要用于**采集数据**：使无人机在特定点附近悬停或飞行，收集状态和控制数据，用于离线 GP 训练。

*   **`comparative_experiment.py`**:
    *   对比实验脚本。用于比较 "无 GP"、"离线 GP" 和 "在线 GP" 等不同控制器的性能。

### 2.4 Visualization (`src/visualization/`)
*   **`plotting.py`**: 绘制 3D 轨迹图、误差曲线等。
*   **`animation.py`**: 生成飞行动画。

## 3. 模块协作关系 (Workflow)

整个系统的工作流可以概括为三个阶段：

### 阶段 1: 数据采集 (Data Collection)
1.  运行 `src/experiments/point_tracking_and_record.py`。
2.  使用基础 MPC (无 GP) 控制无人机飞行。
3.  记录状态 $(x, v, q, \omega)$ 和控制输入 $u$ 到 CSV 文件中。

### 阶段 2: 离线训练 (Offline Training)
1.  运行 `src/gp/offline.py`。
2.  读取阶段 1 采集的数据。
3.  计算残差：$Error = State_{measured} - State_{nominal\_model\_prediction}$。
4.  训练 GP 模型拟合这个 $Error$。
5.  保存训练好的 GP 模型到 `outputs/models/`。

### 阶段 3: GP-MPC 控制 (Deployment/Simulation)
1.  运行 `src/experiments/trajectory_test.py`。
2.  初始化 `Quad3DMPC`。
    *   同时初始化 `Quad3DOptimizer` 和名义动力学模型。
    *   **关键点**: 加载阶段 2 训练好的 GP 模型。
3.  进入控制循环：
    *   **预测**: MPC 在预测未来状态时，不仅使用名义动力学，还加上 GP 预测的误差修正项。
    *   **优化**: 求解出的控制律 $u$ 因此能抵消未知的扰动。
    *   **执行**: 将 $u$ 作用于无人机（或仿真器）。

## 4. 关键文件链接
*   **配置入口**: `config/configuration_parameters.py` (修改仿真参数，如噪声开关)。
*   **GP 配置**: `config/gp_config.py` (定义 GP 的超参数)。
*   **主循环**: 见 `src/experiments/trajectory_test.py` 中的 `main()` 函数。
