"""
Centralized GP configuration for MPC-UAV.

This module provides default configurations for online and offline GP models.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class GPModelParams:
    """
    Standardized parameters for Gaussian Process models.
    """
    length_scale: List[float]  # Length scales for each dimension
    signal_variance: float     # Signal variance (sigma_f^2 or sigma_f)
    noise_variance: float      # Noise variance (sigma_n^2)

@dataclass
class OnlineGPConfig:
    """
    Configuration for online (incremental) Gaussian Process.
    
    Attributes:
        num_dimensions: Number of output dimensions (typically 3 for vx, vy, vz)
        main_process_device: Device for main process ('cpu' or 'cuda')
        worker_device_str: Device for worker processes (must be 'cpu' for multiprocessing)
        buffer_level_capacities: Capacity for each buffer level
        buffer_level_sparsity: Sparsity factor for each buffer level
        min_points_for_initial_train: Minimum points to trigger first training
        min_points_for_ema: Minimum points to enable EMA smoothing
        refit_hyperparams_interval: Number of updates between retraining
        worker_train_iters: Training iterations per worker task
        worker_lr: Learning rate for training
        ema_alpha: EMA smoothing coefficient
    """
    num_dimensions: int = 3
    main_process_device: str = 'cpu'
    worker_device_str: str = 'cpu'
    buffer_level_capacities: List[int] = field(default_factory=lambda: [5, 10, 6])
    buffer_level_sparsity: List[int] = field(default_factory=lambda: [2, 4, 6])
    min_points_for_initial_train: int = 15
    min_points_for_ema: int = 15
    refit_hyperparams_interval: int = 10
    worker_train_iters: int = 20
    worker_lr: float = 0.045
    ema_alpha: float = 0.05
    
    def to_dict(self):
        """Convert config to dictionary for compatibility with existing code."""
        return {
            'num_dimensions': self.num_dimensions,
            'main_process_device': self.main_process_device,
            'worker_device_str': self.worker_device_str,
            'buffer_level_capacities': self.buffer_level_capacities,
            'buffer_level_sparsity': self.buffer_level_sparsity,
            'min_points_for_initial_train': self.min_points_for_initial_train,
            'min_points_for_ema': self.min_points_for_ema,
            'refit_hyperparams_interval': self.refit_hyperparams_interval,
            'worker_train_iters': self.worker_train_iters,
            'worker_lr': self.worker_lr,
            'ema_alpha': self.ema_alpha,
        }


@dataclass  
class OfflineGPConfig:
    """
    Configuration for offline Gaussian Process training.
    
    Attributes:
        x_features: State features to use as input
        u_features: Control features to use as input
        y_features: Output dimensions to predict
        n_train_points: Number of training points per cluster
        n_restarts: Number of optimization restarts
        n_clusters: Number of clusters for ensemble
        histogram_bins: Bins for histogram pruning
        histogram_threshold: Threshold for histogram pruning
        velocity_cap: Maximum velocity for dataset filtering
    """
    x_features: List[int] = field(default_factory=lambda: [7, 8, 9])  # vx, vy, vz
    u_features: List[int] = field(default_factory=list)
    y_features: List[int] = field(default_factory=lambda: [7, 8, 9])
    n_train_points: int = 50
    n_restarts: int = 10
    n_clusters: int = 1
    histogram_bins: int = 10
    histogram_threshold: int = 80
    velocity_cap: float = 0.5


# Default instances for convenience
DEFAULT_ONLINE_GP_CONFIG = OnlineGPConfig()
DEFAULT_OFFLINE_GP_CONFIG = OfflineGPConfig()
