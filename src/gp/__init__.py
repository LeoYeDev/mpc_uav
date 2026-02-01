"""
Gaussian Process module for MPC-UAV.

Contains:
- base: GP implementation (GPEnsemble, CustomGPRegression, CustomKernelFunctions)
- utils: GP utilities (GPDataset, restore_gp_regressors, etc.)
- offline: Offline GP training
- online: Online incremental GP
- rdrv: RDRV model fitting
"""

from src.gp.base import GPEnsemble, CustomGPRegression, CustomKernelFunctions
from src.gp.utils import GPDataset, restore_gp_regressors, read_dataset, world_to_body_velocity_mapping
from src.gp.online import IncrementalGPManager

__all__ = [
    'GPEnsemble', 
    'CustomGPRegression', 
    'CustomKernelFunctions',
    'GPDataset', 
    'restore_gp_regressors', 
    'read_dataset',
    'world_to_body_velocity_mapping',
    'IncrementalGPManager',
]
