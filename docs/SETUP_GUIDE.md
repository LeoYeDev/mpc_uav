# MPC-UAV 环境配置指南

## 快速开始

```bash
conda activate mpc_uav
python src/experiments/comparative_experiment.py
```

---

## 环境配置（已完成）

### Conda 环境 `mpc_uav`

| 包 | 版本 |
|----|------|
| Python | 3.8 |
| casadi | 3.7.2 |
| torch | 2.4.1 (CPU/CUDA) |
| gpytorch | 1.13 |
| acados_template | 0.5.1 |

### ACADOS 安装位置

```
$HOME/acados/           # 主程序
$HOME/acados/lib/       # 库文件
$HOME/acados/bin/       # t_renderer
```

### 环境变量（自动配置）

激活conda环境时自动设置:
- `LD_LIBRARY_PATH` → `$HOME/acados/lib`
- `ACADOS_SOURCE_DIR` → `$HOME/acados`
- `PYTHONPATH` → `/home/jackie/mpc_uav`

---

## GP 模型

| 模型 | 路径 |
|------|------|
| 离线GP | `results/model_fitting/89954f3/simple_sim_gp/` |
| 训练数据 | `data/simplified_sim_dataset/train/dataset_001.csv` |

---

## 常用命令

```bash
# 对比实验（AR-MPC vs Nominal vs SGP-MPC）
python src/experiments/comparative_experiment.py

# 简单轨迹测试
python src/experiments/trajectory_test.py

# 收集新数据
python src/experiments/point_tracking_and_record.py

# 训练GP模型
python src/model_fitting/gp_fitting.py
```

---

## 故障排查

### ACADOS 库找不到
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/acados/lib
```

### 模块导入错误
```bash
export PYTHONPATH=$PYTHONPATH:/home/jackie/mpc_uav
```

### 字体警告
非关键警告，可忽略或安装 Microsoft 字体

---

## 参考

- [ACADOS 文档](https://docs.acados.org/)
- [原始项目](https://github.com/LeoYeDev/data_driven_mpc)
