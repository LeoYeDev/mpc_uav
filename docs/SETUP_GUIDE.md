# MPC-UAV 环境配置指南

本指南详细介绍如何从零开始配置 MPC-UAV 项目的运行环境。

## 系统要求

- **操作系统**: Ubuntu 20.04+ (推荐)
- **Python**: 3.8+
- **编译工具**: GCC, CMake, Git

---

## 1. 系统依赖安装

```bash
sudo apt-get update
sudo apt-get install -y \
    gcc g++ cmake git \
    gnuplot doxygen graphviz \
    libgoogle-glog-dev liblapacke-dev \
    python3-pip python3-venv
```

---

## 2. Python 虚拟环境

```bash
# 安装 virtualenv
pip3 install virtualenv

# 创建虚拟环境
cd ~
virtualenv mpc_venv --python=python3.8

# 激活虚拟环境
source ~/mpc_venv/bin/activate
```

> 💡 每次使用项目前需要激活虚拟环境: `source ~/mpc_venv/bin/activate`

---

## 3. ACADOS 安装

ACADOS 是项目使用的非线性 MPC 求解器。

### 3.1 下载并编译

```bash
# 克隆仓库
cd ~
git clone https://github.com/acados/acados.git
cd acados

# 初始化子模块
git submodule update --recursive --init

# 编译
mkdir -p build && cd build
cmake -DACADOS_WITH_QPOASES=ON ..
make install -j$(nproc)
```

### 3.2 安装 Python 接口

```bash
# 确保虚拟环境已激活
source ~/mpc_venv/bin/activate

# 安装 Python 接口
pip install -e ~/acados/interfaces/acados_template
```

### 3.3 配置环境变量

添加到 `~/.bashrc`:

```bash
# ACADOS
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"$HOME/acados/lib"
export ACADOS_SOURCE_DIR="$HOME/acados"
```

使配置生效:
```bash
source ~/.bashrc
```

### 3.4 安装 Tera Renderer (如需要)

首次运行时如果提示需要 Tera Renderer:

```bash
# 下载
wget https://github.com/acados/tera_renderer/releases/download/v0.0.34/t_renderer-v0.0.34-linux \
    -O ~/acados/bin/t_renderer

# 添加执行权限
chmod +x ~/acados/bin/t_renderer
```

### 3.5 验证 ACADOS 安装

```bash
source ~/mpc_venv/bin/activate
cd ~/acados/examples/acados_python/getting_started/
python minimal_example_ocp.py
```

成功后会显示求解结果图表。

---

## 4. 项目配置

### 4.1 安装项目依赖

```bash
source ~/mpc_venv/bin/activate
cd /path/to/mpc_uav

pip install -r requirements.txt
```

### 4.2 设置 PYTHONPATH

```bash
# 临时设置
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 或添加到 ~/.bashrc (永久)
echo 'export PYTHONPATH=$PYTHONPATH:/path/to/mpc_uav' >> ~/.bashrc
```

---

## 5. 验证安装

### 5.1 核心导入测试

```bash
python -c "
from src.quad_mpc.quad_3d import Quadrotor3D
from src.quad_mpc.quad_3d_mpc import Quad3DMPC
from src.quad_mpc.quad_3d_optimizer import Quad3DOptimizer
print('✓ 核心模块导入成功!')
"
```

### 5.2 轨迹跟踪测试

```bash
python src/experiments/trajectory_test.py
```

预期输出:
```
:::::::::::::: SIMULATION SETUP ::::::::::::::

Simulation: Applied disturbances: 
{"noisy": true, "drag": true, "payload": false, "motor_noise": true}

Model: No regression model loaded

Reference: Executed trajectory `loop` with a peak axial velocity of 8 m/s

::::::::::::: SIMULATION RESULTS :::::::::::::

Mean optimization time: 1.x ms
Tracking RMSE: 0.2xxx m
```

### 5.3 GP 模块测试

```bash
python -m src.model_fitting.test_gp
```

---

## 常见问题

### ACADOS 找不到库文件

确保环境变量已设置:
```bash
echo $LD_LIBRARY_PATH
echo $ACADOS_SOURCE_DIR
```

### CasADi 版本冲突

使用指定版本:
```bash
pip install casadi==3.5.1
```

### 导入错误: No module named 'src'

确保 PYTHONPATH 已正确设置并在项目根目录运行。

---

## 参考资料

- [ACADOS 官方文档](https://docs.acados.org/)
- [原始项目 LeoYeDev/data_driven_mpc](https://github.com/LeoYeDev/data_driven_mpc)
- [论文: Data-Driven MPC for Quadrotors](https://ieeexplore.ieee.org/document/9361343)
