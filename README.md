# MPC-UAV: Gaussian Process Augmented Model Predictive Control

基于高斯过程 (GP) 增强的模型预测控制 (MPC) 四旋翼轨迹跟踪系统。

## 项目简介

本项目实现了一种数据驱动的 MPC 控制方法，通过 GP 回归学习四旋翼的残差动力学，提高高速飞行时的轨迹跟踪精度。

### 核心特性

- **GP-MPC 集成**: 将 GP 均值预测嵌入 ACADOS 非线性 MPC 约束
- **在线学习**: 支持运行时残差学习和模型自适应
- **轨迹生成**: Minimum Snap 轨迹生成器（圆形、Lemniscate、随机轨迹）
- **Simplified Simulation**: 纯 Python 仿真环境，无需 ROS

### 基于项目

本项目基于 [LeoYeDev/data_driven_mpc](https://github.com/LeoYeDev/data_driven_mpc) 改进开发。

---

## 快速开始

### 1. 环境配置

详细配置步骤请参考 [docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md)。

```bash
# 创建虚拟环境
virtualenv mpc_venv --python=python3.8
source mpc_venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 设置 PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

> **注意**: 需要先安装 ACADOS，详见 [SETUP_GUIDE](docs/SETUP_GUIDE.md)。

### 2. 验证安装

```bash
python src/experiments/trajectory_test.py
```

预期输出：
```
::::::::::::: SIMULATION RESULTS :::::::::::::
Mean optimization time: ~1-2 ms
Tracking RMSE: ~0.2 m
```

---

## 项目结构

```
mpc_uav/
├── src/
│   ├── quad_mpc/          # MPC 控制器
│   │   ├── quad_3d.py           # 四旋翼动力学
│   │   ├── quad_3d_mpc.py       # GP-MPC 控制器
│   │   └── quad_3d_optimizer.py # ACADOS 优化器
│   ├── model_fitting/     # GP 模型训练
│   │   ├── gp.py                # 自定义 GP
│   │   ├── gp_fitting.py        # 模型训练
│   │   └── gp_online.py         # 在线学习
│   ├── experiments/       # 实验脚本
│   │   ├── trajectory_test.py
│   │   └── comparative_experiment.py
│   └── utils/             # 工具函数
│       └── trajectories.py      # 轨迹生成
├── config/                # 配置参数
├── docs/                  # 文档
└── requirements.txt
```

---

## 使用示例

### 轨迹跟踪测试

```bash
# 使用圆形轨迹
python src/experiments/trajectory_test.py --trajectory loop --v_max 8

# 使用 Lemniscate 轨迹
python src/experiments/trajectory_test.py --trajectory lemniscate --v_max 6
```

### GP 模型训练

```bash
# 1. 数据采集
python src/experiments/point_tracking_and_record.py --recording --dataset_name my_dataset --simulation_time 300

# 2. 训练 GP 模型
python src/model_fitting/gp_fitting.py --n_points 20 --model_name my_gp --x 7 --y 7
python src/model_fitting/gp_fitting.py --n_points 20 --model_name my_gp --x 8 --y 8
python src/model_fitting/gp_fitting.py --n_points 20 --model_name my_gp --x 9 --y 9
```

### 模型对比实验

```bash
python src/experiments/comparative_experiment.py --model_version <git_hash> --model_name my_gp --model_type gp --fast
```

---

## 配置参数

编辑 `config/configuration_parameters.py`:

```python
class SimpleSimConfig:
    custom_sim_gui = True      # 实时可视化
    result_plots = True        # 结果图表
    
    simulation_disturbances = {
        "noisy": True,         # 推力/力矩噪声
        "drag": True,          # 气动阻力
        "payload": False,      # 载荷扰动
        "motor_noise": True    # 电机噪声
    }
```

---

## License

GPLv3 - 详见 [LICENSE](LICENSE)
