"""
Core module for MPC-UAV quadrotor control.

Contains:
- dynamics: Quadrotor3D dynamics model
- controller: Quad3DMPC controller
- optimizer: Quad3DOptimizer ACADOS interface
"""

from src.core.dynamics import Quadrotor3D
from src.core.controller import Quad3DMPC
from src.core.optimizer import Quad3DOptimizer

__all__ = ['Quadrotor3D', 'Quad3DMPC', 'Quad3DOptimizer']
