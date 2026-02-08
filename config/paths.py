import os

"""
Centralized configuration for project paths and constants.
项目路径和常量的集中配置。
"""

# Project Root Directory (calculated relative to this file: config/paths.py -> config/ -> mpc_uav/)
# 项目根目录 (相对于此文件计算: config/paths.py -> config/ -> mpc_uav/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Directory for Acados generated C code and models
# Acados 生成的 C 代码和模型的目录
ACADOS_MODELS_DIR = os.path.join(PROJECT_ROOT, "acados_models")

# Output directory for results and models
# 结果和模型的输出目录
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

# Default Model Configuration
# 默认模型配置
# These were previously hardcoded in experiments/comparative_experiment.py
# 这些参数此前硬编码在 experiments/comparative_experiment.py 中
DEFAULT_MODEL_VERSION = '89954f3'  # Git hash
DEFAULT_MODEL_NAME = 'simple_sim_gp'
DEFAULT_MODEL_TYPE = 'gp'

# Ensure directories exist
# 确保存储目录存在
os.makedirs(ACADOS_MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
