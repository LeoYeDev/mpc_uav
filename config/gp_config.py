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
    mean: float = 0.0          # Constant mean offset

@dataclass
class OnlineGPConfig:
    """
    Configuration for online (incremental) Gaussian Process.
    
    Attributes:
        num_dimensions: Number of output dimensions (typically 3 for vx, vy, vz)
        main_process_device: Device for main process ('cpu' or 'cuda')
        worker_device_str: Device for worker processes (must be 'cpu' for multiprocessing)
        buffer_max_size: Maximum number of points in buffer
        novelty_weight: Weight for novelty vs recency in buffer scoring (0-1)
        error_threshold: Prediction error threshold for triggering retraining (m/s^2)
        min_points_for_initial_train: Minimum points to trigger first training
        refit_hyperparams_interval: Number of updates between retraining
        worker_train_iters: Training iterations per worker task
        worker_lr: Learning rate for training
    """
    num_dimensions: int = 3
    main_process_device: str = 'cpu'
    worker_device_str: str = 'cpu'
    buffer_max_size: int = 30
    novelty_weight: float = 0.7
    error_threshold: float = 0.15
    min_points_for_initial_train: int = 15
    refit_hyperparams_interval: int = 10
    worker_train_iters: int = 20
    worker_lr: float = 0.045
    
    def to_dict(self):
        """Convert config to dictionary for compatibility with existing code."""
        return {
            'num_dimensions': self.num_dimensions,
            'main_process_device': self.main_process_device,
            'worker_device_str': self.worker_device_str,
            'buffer_max_size': self.buffer_max_size,
            'novelty_weight': self.novelty_weight,
            'error_threshold': self.error_threshold,
            'min_points_for_initial_train': self.min_points_for_initial_train,
            'refit_hyperparams_interval': self.refit_hyperparams_interval,
            'worker_train_iters': self.worker_train_iters,
            'worker_lr': self.worker_lr,
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
