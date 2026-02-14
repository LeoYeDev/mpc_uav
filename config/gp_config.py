"""
Centralized GP configuration for MPC-UAV.
MPC-UAV 的集中式 GP 配置。

本文件统一管理在线/离线 GP 的默认参数，避免不同实验脚本之间出现参数漂移。
"""

from dataclasses import dataclass, field, replace
from typing import List


@dataclass
class GPModelParams:
    """
    Standardized parameters for Gaussian Process models.
    高斯过程模型的标准化参数。
    """

    length_scale: List[float]  # 各输入维度长度尺度
    signal_variance: float  # 信号方差
    noise_variance: float  # 噪声方差
    mean: float = 0.0  # 常值均值项


@dataclass
class OnlineGPConfig:
    """
    在线（增量）GP配置。

    说明：
    1) 该配置会被 `IncrementalGPManager` 直接消费。
    2) 所有实验脚本应通过 `build_online_gp_config()` 构造，避免脚本各自硬编码。
    """

    # =============================
    # 设备与维度
    # =============================
    num_dimensions: int = 3  # 输出维度（通常为机体系 ax/ay/az 残差）
    main_process_device: str = "cpu"  # 主进程推理设备
    worker_device_str: str = "cpu"  # 后台训练设备（多进程建议CPU）

    # =============================
    # 缓冲区基础参数
    # =============================
    buffer_max_size: int = 13  # 训练集目标上限N（C7调优结果：兼顾RMSE与时延）
    novelty_weight: float = 0.55  # 新颖性权重（提升稀疏区覆盖，降低外推误差）
    recency_weight: float = 0.45  # 时效性权重（与新颖性配平，避免过度追新）
    recency_decay_rate: float = 0.10  # 时效性指数衰减率（减缓旧点失效，提升稳态拟合）
    buffer_min_distance: float = 0.02  # 速度邻域半径（去重/密度评估）

    # =============================
    # 多级缓存结构参数
    # =============================
    buffer_level_capacities: List[int] = field(default_factory=lambda: [12, 5, 3])
    buffer_level_sparsity: List[int] = field(default_factory=lambda: [1, 2, 5])

    # =============================
    # IVS评分与聚簇平滑参数
    # =============================
    cluster_anchor_window: int = 6  # 主簇锚点窗口（从L0最近点中取中位）
    cluster_gap_factor: float = 2.5  # 切簇间距倍率
    out_cluster_penalty: float = 0.08  # 非主簇惩罚强度（温和）
    target_size_slack: int = 2  # 训练集弹性下界：N-slack..N（适度放松容量以换时延）

    # =============================
    # 近邻快速更新与局部重复控制
    # =============================
    buffer_local_dup_cap: int = 2  # 同一速度邻域允许的最大重复样本数
    buffer_close_update_v_ratio: float = 0.3  # 近邻覆盖速度阈值系数
    buffer_close_update_y_threshold: float = 0.3  # 近邻覆盖残差阈值

    # =============================
    # IVS全量/增量刷新控制
    # =============================
    ivs_full_rescore_period: int = 4  # 每插入多少次触发一次全量重评分（降低重评分频率）
    ivs_coverage_bins: int = 10  # 覆盖优先分箱数
    ivs_min_cover_ratio: float = 0.55  # 覆盖优先最小比例

    # =============================
    # 在线预测性能参数
    # =============================
    ivs_query_clamp_margin: float = 0.10  # 查询软夹紧边界比例
    online_predict_need_variance_when_alpha_zero: bool = False  # alpha=0时默认跳过方差
    force_cpu_predict: bool = True  # 强制CPU推理，降低GPU初始化/切换开销
    predict_use_likelihood_variance: bool = False  # 方差是否包含观测噪声项
    predict_cache_enabled: bool = True  # 启用查询缓存
    predict_cache_tolerance: float = 0.16  # 查询缓存触发阈值（在不明显伤精度前提下进一步提高命中率）
    torch_num_threads: int = 1  # Torch线程数（小模型通常1更稳）

    # =============================
    # 在线训练触发参数
    # =============================
    error_threshold: float = 0.14  # 误差触发阈值（减小线程训练抖动）
    min_points_for_initial_train: int = 9  # 首次训练最小样本数（避免大于缓冲上限导致强制截断）
    refit_hyperparams_interval: int = 24  # 定期重训间隔（按数据更新计）
    worker_train_iters: int = 8  # 后台训练迭代数（降低控制回路受后台训练干扰）
    worker_lr: float = 0.045  # 后台训练学习率

    # =============================
    # 核函数参数
    # =============================
    gp_kernel: str = "rbf"  # rbf/matern12/matern32/matern52/matern_nu
    gp_matern_nu: float = 2.5  # matern_nu 时使用

    # =============================
    # 混合门控降时延（新增）
    # =============================
    online_update_stride: int = 2  # 每隔多少步检查一次训练触发（2=降时延且精度可接受）

    # =============================
    # 消融开关
    # =============================
    buffer_type: str = "ivs"  # ivs/fifo
    async_hp_updates: bool = True  # 异步后台训练
    variance_scaling_alpha: float = 1.0  # MPC风险缩放项

    def to_dict(self):
        """转为字典，供现有模块兼容读取。"""

        return {
            "num_dimensions": self.num_dimensions,
            "main_process_device": self.main_process_device,
            "worker_device_str": self.worker_device_str,
            "buffer_max_size": self.buffer_max_size,
            "novelty_weight": self.novelty_weight,
            "recency_weight": self.recency_weight,
            "recency_decay_rate": self.recency_decay_rate,
            "buffer_min_distance": self.buffer_min_distance,
            "buffer_level_capacities": self.buffer_level_capacities,
            "buffer_level_sparsity": self.buffer_level_sparsity,
            "cluster_anchor_window": self.cluster_anchor_window,
            "cluster_gap_factor": self.cluster_gap_factor,
            "out_cluster_penalty": self.out_cluster_penalty,
            "target_size_slack": self.target_size_slack,
            "buffer_local_dup_cap": self.buffer_local_dup_cap,
            "buffer_close_update_v_ratio": self.buffer_close_update_v_ratio,
            "buffer_close_update_y_threshold": self.buffer_close_update_y_threshold,
            "ivs_full_rescore_period": self.ivs_full_rescore_period,
            "ivs_coverage_bins": self.ivs_coverage_bins,
            "ivs_min_cover_ratio": self.ivs_min_cover_ratio,
            "ivs_query_clamp_margin": self.ivs_query_clamp_margin,
            "online_predict_need_variance_when_alpha_zero": self.online_predict_need_variance_when_alpha_zero,
            "force_cpu_predict": self.force_cpu_predict,
            "predict_use_likelihood_variance": self.predict_use_likelihood_variance,
            "predict_cache_enabled": self.predict_cache_enabled,
            "predict_cache_tolerance": self.predict_cache_tolerance,
            "torch_num_threads": self.torch_num_threads,
            "error_threshold": self.error_threshold,
            "min_points_for_initial_train": self.min_points_for_initial_train,
            "refit_hyperparams_interval": self.refit_hyperparams_interval,
            "worker_train_iters": self.worker_train_iters,
            "worker_lr": self.worker_lr,
            "gp_kernel": self.gp_kernel,
            "gp_matern_nu": self.gp_matern_nu,
            "online_update_stride": self.online_update_stride,
            "buffer_type": self.buffer_type,
            "async_hp_updates": self.async_hp_updates,
            "variance_scaling_alpha": self.variance_scaling_alpha,
        }


@dataclass
class OfflineGPConfig:
    """
    离线 GP 训练配置。
    """

    x_features: List[int] = field(default_factory=lambda: [7, 8, 9])  # 输入特征索引
    u_features: List[int] = field(default_factory=list)  # 控制输入特征索引
    y_features: List[int] = field(default_factory=lambda: [7, 8, 9])  # 输出特征索引
    n_train_points: int = 50  # 每簇训练点数
    n_restarts: int = 10  # 超参数优化重启次数
    n_clusters: int = 1  # 聚类数
    histogram_bins: int = 10  # 直方图分箱数
    histogram_threshold: int = 80  # 直方图阈值
    velocity_cap: float = 0.5  # 数据筛选速度上限


# 默认实例
DEFAULT_ONLINE_GP_CONFIG = OnlineGPConfig()
DEFAULT_OFFLINE_GP_CONFIG = OfflineGPConfig()


def build_online_gp_config(
    *,
    buffer_type: str = "ivs",
    async_hp_updates: bool = True,
    variance_scaling_alpha: float = 0.0,
    **overrides,
) -> OnlineGPConfig:
    """
    构建统一在线GP配置入口：
    1) 总是从 DEFAULT_ONLINE_GP_CONFIG 派生。
    2) 仅通过显式 overrides 改动差异参数。
    """

    cfg = replace(
        DEFAULT_ONLINE_GP_CONFIG,
        buffer_type=str(buffer_type),
        async_hp_updates=bool(async_hp_updates),
        variance_scaling_alpha=float(variance_scaling_alpha),
    )
    if overrides:
        cfg = replace(cfg, **overrides)
    return cfg
