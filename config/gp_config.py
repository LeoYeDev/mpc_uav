"""
MPC-UAV 高斯过程配置中心。

说明：
1) 统一在线/离线 GP 的默认参数，避免实验脚本各自硬编码。
2) 当前在线缓冲策略采用“简化版多级 IVS”：
   - 两阈值去密集（入样阈值、旧近邻剔除阈值）
   - 方向反转时按上限删除最旧点
3) 所有实验脚本应通过 build_online_gp_config() 构造配置。
"""

from dataclasses import dataclass, field, replace
from typing import List


@dataclass
class GPModelParams:
    """高斯过程模型参数统一结构。"""

    length_scale: List[float]  # 各输入维度长度尺度
    signal_variance: float  # 信号方差
    noise_variance: float  # 噪声方差
    mean: float = 0.0  # 常值均值项


@dataclass
class OnlineGPConfig:
    """在线（增量）GP配置。"""

    # =============================
    # 设备与维度
    # =============================
    num_dimensions: int = 3  # 输出维度（通常为机体系 ax/ay/az 残差）
    main_process_device: str = "cpu"  # 主进程推理设备
    worker_device_str: str = "cpu"  # 后台训练设备（多进程建议CPU）

    # =============================
    # 缓冲区核心参数（简化版）
    # =============================
    buffer_max_size: int = 15  # 训练集总上限 N（三级总容量严格等于 N）
    buffer_insert_min_delta_v: float = 0.15  # 入样二维门控阈值（(v,y)欧式距离，比较新点与L0最新点）
    buffer_prune_old_delta_v: float = 0.05  # 旧点剔除阈值（仅按横坐标v判断，|v_old-v_new| < 阈值）
    buffer_flip_prune_limit: int = 2  # 单次插入最多循环执行“方向反转删最旧点”的次数
    buffer_level_capacities: List[int] = field(default_factory=lambda: [9, 4, 2])  # 三级缓存容量（总和建议等于 buffer_max_size）
    buffer_level_sparsity: List[int] = field(default_factory=lambda: [1, 2, 5])  # 三级稀疏采样因子（L0/L1/L2）
    novelty_weight: float = 0.55  # 新颖性权重（越大越偏向覆盖稀疏速度区域）
    recency_weight: float = 0.45  # 时效性权重（越大越偏向保留新样本）
    recency_decay_rate: float = 0.15  # 时效性指数衰减系数（越大衰减越快）

    # =============================
    # 在线预测性能参数
    # =============================
    force_cpu_predict: bool = True  # 强制 CPU 推理，避免小模型的 GPU 切换开销
    predict_use_likelihood_variance: bool = False  # 方差是否包含观测噪声项
    predict_cache_enabled: bool = True  # 启用查询缓存
    predict_cache_tolerance: float = 0.16  # 查询缓存触发阈值
    torch_num_threads: int = 1  # Torch 线程数（小模型常用 1）

    # =============================
    # 在线训练触发参数
    # =============================
    error_threshold: float = 14  # 误差触发阈值
    min_points_for_initial_train: int = 10  # 首次训练最小样本数
    refit_hyperparams_interval: int = 15  # 定期重训间隔（按数据更新计）
    online_update_stride: int = 2  # 每隔多少步检查一次训练触发（1=每步）
    worker_train_iters: int = 15  # 后台训练迭代数
    worker_lr: float = 0.045  # 后台训练学习率

    # =============================
    # 核函数参数
    # =============================
    gp_kernel: str = "rbf"  # rbf/matern12/matern32/matern52/matern_nu
    gp_matern_nu: float = 2.5  # matern_nu 时使用

    # =============================
    # 消融与主流程开关
    # =============================
    buffer_type: str = "ivs"  # ivs/fifo
    async_hp_updates: bool = True  # 异步后台训练
    variance_scaling_alpha: float = 1.0  # MPC 风险缩放项

    def to_dict(self) -> dict:
        """转为字典，供现有模块兼容读取。"""
        return {
            "num_dimensions": self.num_dimensions,
            "main_process_device": self.main_process_device,
            "worker_device_str": self.worker_device_str,
            "buffer_max_size": self.buffer_max_size,
            "buffer_insert_min_delta_v": self.buffer_insert_min_delta_v,
            "buffer_prune_old_delta_v": self.buffer_prune_old_delta_v,
            "buffer_flip_prune_limit": self.buffer_flip_prune_limit,
            "buffer_level_capacities": self.buffer_level_capacities,
            "buffer_level_sparsity": self.buffer_level_sparsity,
            "novelty_weight": self.novelty_weight,
            "recency_weight": self.recency_weight,
            "recency_decay_rate": self.recency_decay_rate,
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
    """离线 GP 训练配置。"""

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
    variance_scaling_alpha: float = 1.0,
    **overrides,
) -> OnlineGPConfig:
    """
    构建统一在线 GP 配置入口。

    规则：
    1) 总是从 DEFAULT_ONLINE_GP_CONFIG 派生。
    2) 允许显式覆盖现有字段。
    3) 对历史旧字段做兼容忽略（打印提示，不抛错）。
    """

    cfg = replace(
        DEFAULT_ONLINE_GP_CONFIG,
        buffer_type=str(buffer_type),
        async_hp_updates=bool(async_hp_updates),
        variance_scaling_alpha=float(variance_scaling_alpha),
    )

    if not overrides:
        return cfg

    deprecated_keys = {
        "cluster_anchor_window",
        "cluster_gap_factor",
        "out_cluster_penalty",
        "target_size_slack",
        "ivs_coverage_bins",
        "ivs_min_cover_ratio",
        "ivs_full_rescore_period",
        "ivs_query_clamp_margin",
        "online_predict_need_variance_when_alpha_zero",
        "buffer_min_distance",
        "buffer_local_dup_cap",
        "buffer_close_update_v_ratio",
        "buffer_close_update_y_threshold",
    }

    valid_keys = set(OnlineGPConfig.__dataclass_fields__.keys())
    filtered = {}
    for key, value in overrides.items():
        if key in deprecated_keys:
            print(f"[build_online_gp_config] deprecated override ignored: {key}")
            continue
        if key not in valid_keys:
            raise TypeError(f"Unknown OnlineGPConfig override: {key}")
        filtered[key] = value

    if filtered:
        cfg = replace(cfg, **filtered)
    return cfg
