"""
Centralized GP configuration for MPC-UAV.
MPC-UAV 的集中式 GP 配置。

This module provides default configurations for online and offline GP models.
本模块提供在线和离线 GP 模型的默认配置。
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class GPModelParams:
    """
    Standardized parameters for Gaussian Process models.
    高斯过程模型的标准化参数。
    """
    length_scale: List[float]  # 各维度的长度尺度 (Length scales for each dimension)
    signal_variance: float     # 信号方差 (sigma_f^2 or sigma_f) (Signal variance)
    noise_variance: float      # 噪声方差 (sigma_n^2) (Noise variance)
    mean: float = 0.0          # 常数均值偏移 (Constant mean offset)

@dataclass
class OnlineGPConfig:
    """
    Configuration for online (incremental) Gaussian Process.
    在线（增量）高斯过程的配置。
    
    Attributes:
        num_dimensions: 输出维数（通常为 vx, vy, vz 的 3 维）
        main_process_device: 主进程使用的设备 ('cpu' 或 'cuda')
        worker_device_str: 工作进程使用的设备 (多进程必须使用 'cpu')
        buffer_max_size: 缓冲区中的最大点数
        novelty_weight: 缓冲区评分中新颖性相对于新近性的权重 (0-1)
        error_threshold: 触发重训练的预测误差阈值 (m/s^2)
        min_points_for_initial_train: 触发首次训练所需的最小点数
        refit_hyperparams_interval: 重训练之间的更新次数间隔
        worker_train_iters: 每个工作任务的训练迭代次数
        worker_lr: 训练的学习率
        
        # Ablation switches (消融开关)
        buffer_type: 缓冲区类型 ('ivs' 信息值选择 / 'fifo' 先进先出)
        async_hp_updates: 是否异步更新超参数 (True 异步 / False 同步)
        variance_scaling_alpha: 方差缩放系数 (0=完全信任GP, 1=默认保守)
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
    
    # Ablation switches (消融实验开关)
    buffer_type: str = 'ivs'           # 'ivs' (Information Value Selection) or 'fifo'
    async_hp_updates: bool = True      # True=async, False=sync (blocking)
    variance_scaling_alpha: float = 1.0  # 0=full GP trust, higher=more conservative
    
    def to_dict(self):
        """
        Convert config to dictionary for compatibility with existing code.
        将配置转换为字典以兼容现有代码。
        """
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
            # Ablation switches
            'buffer_type': self.buffer_type,
            'async_hp_updates': self.async_hp_updates,
            'variance_scaling_alpha': self.variance_scaling_alpha,
        }


@dataclass  
class OfflineGPConfig:
    """
    Configuration for offline Gaussian Process training.
    离线高斯过程训练的配置。
    
    Attributes:
        x_features: 用作输入的状态特征索引
                    State features to use as input
        u_features: 用作输入的控制特征索引
                    Control features to use as input
        y_features: 需要预测的输出维度索引
                    Output dimensions to predict
        n_train_points: 每个聚类的训练点数
                        Number of training points per cluster
        n_restarts: 优化重启次数
                    Number of optimization restarts
        n_clusters: 集成的聚类数量
                    Number of clusters for ensemble
        histogram_bins: 直方图剪枝的分箱数
                        Bins for histogram pruning
        histogram_threshold: 直方图剪枝的阈值
                             Threshold for histogram pruning
        velocity_cap: 数据集过滤的最大速度上限
                      Maximum velocity for dataset filtering
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


# 默认实例 (Default instances for convenience)
DEFAULT_ONLINE_GP_CONFIG = OnlineGPConfig()
DEFAULT_OFFLINE_GP_CONFIG = OfflineGPConfig()
