import os

"""
Centralized configuration for project paths and constants.
"""

# Project Root Directory (calculated relative to this file: config/paths.py -> config/ -> mpc_uav/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Directory for Acados generated C code and models
ACADOS_MODELS_DIR = os.path.join(PROJECT_ROOT, "acados_models")

# Output directory for results and models
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

# Default Model Configuration
# These were previously hardcoded in experiments/comparative_experiment.py
DEFAULT_MODEL_VERSION = '89954f3'  # Git hash
DEFAULT_MODEL_NAME = 'simple_sim_gp'
DEFAULT_MODEL_TYPE = 'gp'

# Ensure directories exist
os.makedirs(ACADOS_MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
