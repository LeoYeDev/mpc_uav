
"""
Data buffer strategies for Online GP.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

class BaseBuffer(ABC):
    """Abstract base class for data buffers."""
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.data: List[Tuple[float, float, float]] = []  # List of (v, y, t)
        self.total_adds = 0  # Counter for total data points added (for recency)

    @abstractmethod
    def insert(self, v_scalar: float, y_scalar: float, timestamp: float) -> None:
        """Insert a new data point into the buffer."""
        pass

    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return current buffer data as numpy arrays (X, Y)."""
        if not self.data:
            return np.empty((0, 1)), np.empty((0, 1))
        
        data_arr = np.array(self.data)
        return data_arr[:, 0:1], data_arr[:, 1:2]

    def get_training_set(self) -> List[Tuple[float, float]]:
        """Return list of (v, y) tuples for training."""
        return [(p[0], p[1]) for p in self.data]


    def reset(self):
        """Clear the buffer."""
        self.data = []
        self.total_adds = 0


class FIFOBuffer(BaseBuffer):
    """First-In-First-Out Buffer."""
    
    def insert(self, v_scalar: float, y_scalar: float, timestamp: float) -> None:
        """Insert new data point, automatically removing the oldest if full."""
        if len(self.data) >= self.max_size:
            self.data.pop(0)
        self.data.append((v_scalar, y_scalar, timestamp))
        self.total_adds += 1


class InformationGainBuffer(BaseBuffer):
    """
    Buffer using Information Value Score (IVS) to select data points.
    
    Score = w_novelty * novelty_score + w_recency * recency_score
    
    - Novelty: Based on distance to nearest neighbor in buffer.
               Uses adaptive distance scale to avoid outlier-sensitive normalization.
    - Recency: Exponential decay based on time difference. 
               exp(-decay_rate * (t_current - t_point)) -> [0, 1]
    """
    
    def __init__(self, max_size: int, novelty_weight: float = 0.5,
                 recency_weight: float = 0.5, decay_rate: float = 0.1,
                 min_distance: float = 0.01,
                 local_dup_cap: int = 4,
                 close_update_v_ratio: float = 0.35,
                 close_update_y_threshold: float = 0.03):
        super().__init__(max_size)
        self.novelty_weight = max(float(novelty_weight), 0.0)
        self.recency_weight = max(float(recency_weight), 0.0)
        if self.novelty_weight == 0.0 and self.recency_weight == 0.0:
            self.novelty_weight = 1.0
        total_w = self.novelty_weight + self.recency_weight
        self.novelty_weight /= total_w
        self.recency_weight /= total_w
        self.decay_rate = max(float(decay_rate), 1e-6)
        self.min_distance = max(float(min_distance), 0.0)  # Minimum distance to consider adding
        # 同一速度邻域允许保留的最大样本数。超过后才执行“就地替换”。
        self.local_dup_cap = max(1, int(local_dup_cap))
        # 近邻快速更新阈值：
        # 仅当 (|Δv| <= ratio*min_distance) 且 (|Δy| <= y_threshold) 时直接覆盖最近点。
        self.close_update_v_ratio = float(np.clip(close_update_v_ratio, 0.0, 5.0))
        self.close_update_y_threshold = max(float(close_update_y_threshold), 0.0)

    def _compute_recency_scores(self, timestamps: np.ndarray) -> np.ndarray:
        """
        Compute recency scores with adaptive time scaling.

        Raw timestamps in this project are often separated by ~0.02s, and the
        previous implementation used absolute seconds directly. With the default
        decay, that made recency almost flat and IVS effectively novelty-only.
        """
        if timestamps.size <= 1:
            return np.ones_like(timestamps, dtype=float)

        t_newest = float(np.max(timestamps))
        t_oldest = float(np.min(timestamps))
        unique_ts = np.unique(np.sort(timestamps))

        if unique_ts.size >= 2:
            step_scale = float(np.median(np.diff(unique_ts)))
        else:
            step_scale = t_newest - t_oldest

        step_scale = max(step_scale, 1e-6)
        age_in_steps = (t_newest - timestamps) / step_scale
        return np.exp(-self.decay_rate * age_in_steps)

    def _compute_scores(self) -> np.ndarray:
        """Compute IVS scores for all points in the buffer."""
        n = len(self.data)
        if n == 0:
            return np.array([])
        if n == 1:
            return np.array([1.0])
        
        # Extract data
        velocities = np.array([p[0] for p in self.data], dtype=float)
        timestamps = np.array([p[2] for p in self.data], dtype=float)
        
        # --- Novelty Score ---
        # Novelty = distance novelty * density penalty.
        # This favors under-covered velocity regions and suppresses crowded bins.
        dists = np.abs(velocities[:, None] - velocities[None, :])
        np.fill_diagonal(dists, np.inf)
        min_dists = np.min(dists, axis=1)

        rho = max(float(self.min_distance), 1e-6)
        density = np.sum(dists <= rho, axis=1).astype(float)

        distance_scale = max(float(np.median(min_dists)), rho, 1e-6)
        distance_novelty = 1.0 - np.exp(-min_dists / distance_scale)
        novelty_scores = distance_novelty * (1.0 / (1.0 + density))
            
        # --- Recency Score ---
        # Exponential decay over "sample steps" (adaptive to timestamp scale)
        recency_scores = self._compute_recency_scores(timestamps)
        
        # Combine
        total_scores = self.novelty_weight * novelty_scores + self.recency_weight * recency_scores
        
        return total_scores

    def _find_joint_close_index(self, v_scalar: float, y_scalar: float) -> Optional[int]:
        """
        查找同时在速度轴与残差轴都非常接近的候选点索引。
        命中后可直接原地更新，避免进入完整 IVS 评分流程。
        """
        if not self.data:
            return None
        velocities = np.array([p[0] for p in self.data], dtype=float)
        idx = int(np.argmin(np.abs(velocities - float(v_scalar))))
        v_dist = abs(float(velocities[idx]) - float(v_scalar))
        v_tol = max(float(self.min_distance) * float(self.close_update_v_ratio), 1e-9)
        if v_dist > v_tol:
            return None
        y_dist = abs(float(self.data[idx][1]) - float(y_scalar))
        if y_dist > self.close_update_y_threshold:
            return None
        return idx

    def insert(self, v_scalar: float, y_scalar: float, timestamp: float) -> None:
        """
        Insert new point. If full, remove point with lowest IVS score.
        Also skips insertion if point is too close to existing points (redundant).
        """
        # 快速路径：横纵坐标都极近时，直接覆盖最近点，避免每次都触发全量评分。
        joint_close_idx = self._find_joint_close_index(v_scalar, y_scalar)
        if joint_close_idx is not None:
            self.data[int(joint_close_idx)] = (float(v_scalar), float(y_scalar), float(timestamp))
            self.total_adds += 1
            return

        # 局部重复控制：
        # 仅当同一速度邻域样本数超过 local_dup_cap 时才执行替换。
        # 这样可以避免缓冲区在窄速度区间“塌缩成极少点”。
        if self.data:
            velocities = np.array([p[0] for p in self.data])
            dist_to_existing = np.min(np.abs(velocities - v_scalar))
            if dist_to_existing < self.min_distance:
                close_idx = np.where(np.abs(velocities - v_scalar) < self.min_distance)[0]
                if len(close_idx) >= self.local_dup_cap:
                    # 邻域已满：替换该邻域中最旧样本，保留时效性。
                    oldest_idx = int(min(close_idx, key=lambda i: self.data[int(i)][2]))
                    self.data[oldest_idx] = (float(v_scalar), float(y_scalar), float(timestamp))
                    self.total_adds += 1
                    return

        if len(self.data) < self.max_size:
            self.data.append((float(v_scalar), float(y_scalar), float(timestamp)))
        else:
            # Buffer full: keep N points with highest combined novelty+recency.
            self.data.append((float(v_scalar), float(y_scalar), float(timestamp)))
            scores = self._compute_scores()
            
            # Find index to remove
            idx_to_remove = np.argmin(scores)
            
            # Remove
            self.data.pop(idx_to_remove)
            
        self.total_adds += 1


class MultiLevelInformationGainBuffer(BaseBuffer):
    """
    Multi-level IVS buffer with unified scoring and cluster-consistency smoothing.

    Design goals:
    - Keep short/mid/long-term memory with different sampling strides.
    - Use one single IVS formula across all levels and final merge.
    - Encourage sparse coverage in velocity space while preserving recency.
    - Penalize stale out-of-cluster samples to keep a smooth evolving data manifold.
    - Keep final training size close to FIFO fairness (N-target_size_slack .. N).
    """

    def __init__(
        self,
        max_size: int,
        novelty_weight: float = 0.5,
        recency_weight: float = 0.5,
        decay_rate: float = 0.1,
        min_distance: float = 0.01,
        level_capacities: Optional[List[int]] = None,
        level_sparsity: Optional[List[int]] = None,
        merge_min_distance: Optional[float] = None,
        cluster_anchor_window: int = 6,
        cluster_gap_factor: float = 2.5,
        out_cluster_penalty: float = 0.35,
        target_size_slack: int = 1,
        local_dup_cap: int = 4,
        close_update_v_ratio: float = 0.35,
        close_update_y_threshold: float = 0.03,
    ):
        super().__init__(max_size)
        self.novelty_weight = max(float(novelty_weight), 0.0)
        self.recency_weight = max(float(recency_weight), 0.0)
        if self.novelty_weight == 0.0 and self.recency_weight == 0.0:
            self.recency_weight = 1.0
        total_w = self.novelty_weight + self.recency_weight
        self.novelty_weight /= total_w
        self.recency_weight /= total_w
        self.decay_rate = max(float(decay_rate), 1e-6)
        self.min_distance = max(float(min_distance), 0.0)
        # Deprecated compatibility field. Merge ratio/min-distance is no longer
        # used for final selection logic.
        self.merge_min_distance = (
            max(float(merge_min_distance), 0.0)
            if merge_min_distance is not None
            else self.min_distance
        )
        self.cluster_anchor_window = max(1, int(cluster_anchor_window))
        self.cluster_gap_factor = max(1.0, float(cluster_gap_factor))
        self.out_cluster_penalty = float(np.clip(out_cluster_penalty, 0.0, 0.95))
        self.target_size_slack = max(0, int(target_size_slack))
        self.local_dup_cap = max(1, int(local_dup_cap))
        self.close_update_v_ratio = float(np.clip(close_update_v_ratio, 0.0, 5.0))
        self.close_update_y_threshold = max(float(close_update_y_threshold), 0.0)

        self.level_capacities = self._resolve_level_capacities(max_size, level_capacities)
        self.level_sparsity = self._resolve_level_sparsity(
            n_levels=len(self.level_capacities),
            level_sparsity=level_sparsity,
        )
        self.levels = self._build_levels()

        self._dirty = True
        self._cached_training_set: List[Tuple[float, float, float]] = []
        self._last_main_cluster_ratio: float = 1.0
        self._last_selected_size: int = 0

    def _resolve_level_capacities(self, max_size: int, capacities: Optional[List[int]]) -> List[int]:
        if capacities:
            caps = [int(c) for c in capacities if int(c) > 0]
        else:
            # Default 3 levels: recent / medium / long-term memory.
            c0 = max(1, int(round(max_size * 0.5)))
            c1 = max(1, int(round(max_size * 0.3)))
            c2 = max(1, max_size - c0 - c1)
            caps = [c0, c1, c2]

        if not caps:
            caps = [max_size]

        total = sum(caps)
        if total < max_size:
            caps[0] += (max_size - total)
        elif total > max_size:
            # Shrink proportionally, keep each level >= 1.
            scale = max_size / float(total)
            caps = [max(1, int(round(c * scale))) for c in caps]
            while sum(caps) > max_size:
                idx = int(np.argmax(caps))
                if caps[idx] > 1:
                    caps[idx] -= 1
                else:
                    break
            while sum(caps) < max_size:
                caps[0] += 1
        return caps

    def _resolve_level_sparsity(self, n_levels: int, level_sparsity: Optional[List[int]]) -> List[int]:
        if level_sparsity:
            sparse = [max(1, int(s)) for s in level_sparsity]
        else:
            sparse = [1, 2, 5]

        if len(sparse) < n_levels:
            last = sparse[-1] if sparse else 1
            for _ in range(n_levels - len(sparse)):
                last = max(1, last + 3)
                sparse.append(last)
        return sparse[:n_levels]

    def _build_levels(self) -> List[InformationGainBuffer]:
        # Unified IVS score for all levels. Only capacity/stride differs.
        levels: List[InformationGainBuffer] = []
        for cap in self.level_capacities:
            levels.append(
                InformationGainBuffer(
                    max_size=cap,
                    novelty_weight=self.novelty_weight,
                    recency_weight=self.recency_weight,
                    decay_rate=self.decay_rate,
                    min_distance=self.min_distance,
                    local_dup_cap=self.local_dup_cap,
                    close_update_v_ratio=self.close_update_v_ratio,
                    close_update_y_threshold=self.close_update_y_threshold,
                )
            )
        return levels

    def insert(self, v_scalar: float, y_scalar: float, timestamp: float) -> None:
        v = float(v_scalar)
        y = float(y_scalar)
        t = float(timestamp)

        # Each level uses the same IVS logic. Levels differ only in temporal stride.
        for level_idx, level in enumerate(self.levels):
            stride = self.level_sparsity[level_idx]
            if self.total_adds % stride == 0:
                level.insert(v, y, t)

        self.total_adds += 1
        self._dirty = True

    def _compute_recency_scores(self, timestamps: np.ndarray) -> np.ndarray:
        if timestamps.size <= 1:
            return np.ones_like(timestamps, dtype=float)
        t_newest = float(np.max(timestamps))
        unique_ts = np.unique(np.sort(timestamps))
        if unique_ts.size >= 2:
            step_scale = float(np.median(np.diff(unique_ts)))
        else:
            step_scale = 1.0
        step_scale = max(step_scale, 1e-6)
        age_in_steps = (t_newest - timestamps) / step_scale
        return np.exp(-self.decay_rate * age_in_steps)

    def _compute_unified_scores(
        self,
        velocities: np.ndarray,
        timestamps: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = velocities.size
        if n == 0:
            empty = np.array([], dtype=float)
            return empty, empty, empty
        if n == 1:
            one = np.ones(1, dtype=float)
            return one, one, one

        dists = np.abs(velocities[:, None] - velocities[None, :])
        np.fill_diagonal(dists, np.inf)
        min_dists = np.min(dists, axis=1)

        rho = max(float(self.min_distance), 1e-6)
        density = np.sum(dists <= rho, axis=1).astype(float)
        distance_scale = max(float(np.median(min_dists)), rho, 1e-6)
        distance_novelty = 1.0 - np.exp(-min_dists / distance_scale)
        novelty = distance_novelty * (1.0 / (1.0 + density))

        recency = self._compute_recency_scores(timestamps)
        total = self.novelty_weight * novelty + self.recency_weight * recency
        return total, novelty, recency

    def _get_anchor_velocity(self) -> Optional[float]:
        if not self.levels:
            return None
        level0 = self.levels[0].data
        if not level0:
            return None
        k = min(self.cluster_anchor_window, len(level0))
        recent = sorted(level0, key=lambda p: p[2])[-k:]
        return float(np.median([p[0] for p in recent]))

    def _identify_main_cluster(
        self,
        velocities: np.ndarray,
        timestamps: np.ndarray,
    ) -> Tuple[np.ndarray, int, float]:
        n = velocities.size
        if n == 0:
            return np.array([], dtype=int), -1, 0.0
        if n == 1:
            return np.zeros(1, dtype=int), 0, 1.0

        sort_idx = np.argsort(velocities)
        sorted_v = velocities[sort_idx]
        diffs = np.diff(sorted_v)
        finite_diffs = diffs[np.isfinite(diffs)]
        if finite_diffs.size:
            local_scale = max(float(np.median(np.abs(finite_diffs))), float(self.min_distance), 1e-6)
        else:
            local_scale = max(float(self.min_distance), 1e-6)
        gap_threshold = self.cluster_gap_factor * local_scale

        labels_sorted = np.zeros(n, dtype=int)
        cluster_id = 0
        for i, d in enumerate(diffs):
            if d > gap_threshold:
                cluster_id += 1
            labels_sorted[i + 1] = cluster_id

        labels = np.empty(n, dtype=int)
        labels[sort_idx] = labels_sorted

        anchor_v = self._get_anchor_velocity()
        if anchor_v is None:
            anchor_idx = int(np.argmax(timestamps))
        else:
            anchor_idx = int(np.argmin(np.abs(velocities - anchor_v)))
        main_label = int(labels[anchor_idx])
        main_ratio = float(np.mean(labels == main_label))
        return labels, main_label, main_ratio

    def _compute_merged_points(self) -> List[Tuple[float, float, float]]:
        candidates: List[Tuple[float, float, float]] = []
        for level in self.levels:
            for v, y, t in level.data:
                candidates.append((float(v), float(y), float(t)))

        if not candidates:
            self._last_main_cluster_ratio = 0.0
            self._last_selected_size = 0
            return []

        velocities = np.array([p[0] for p in candidates], dtype=float)
        timestamps = np.array([p[2] for p in candidates], dtype=float)
        total_scores, novelty_scores, recency_scores = self._compute_unified_scores(velocities, timestamps)

        labels, main_label, main_ratio = self._identify_main_cluster(velocities, timestamps)
        penalized_scores = total_scores.copy()
        if main_label >= 0:
            penalized_scores[labels != main_label] *= (1.0 - self.out_cluster_penalty)

        order = sorted(
            range(len(candidates)),
            key=lambda i: (
                penalized_scores[i],
                recency_scores[i],
                novelty_scores[i],
                timestamps[i],
            ),
            reverse=True,
        )

        lower_target = max(1, int(self.max_size) - int(self.target_size_slack))
        upper_target = int(self.max_size)
        rho = max(float(self.min_distance), 1e-6)

        selected: List[int] = []
        selected_set = set()

        for idx in order:
            if len(selected) >= upper_target:
                break
            if not selected:
                selected.append(idx)
                selected_set.add(idx)
                continue
            d_min = float(np.min(np.abs(velocities[idx] - velocities[np.array(selected, dtype=int)])))
            if d_min >= rho:
                selected.append(idx)
                selected_set.add(idx)

        if len(selected) < lower_target:
            backfill_order = sorted(
                [i for i in range(len(candidates)) if i not in selected_set],
                key=lambda i: (recency_scores[i], penalized_scores[i], timestamps[i]),
                reverse=True,
            )
            for idx in backfill_order:
                selected.append(idx)
                selected_set.add(idx)
                if len(selected) >= lower_target:
                    break

        if len(selected) < upper_target:
            relaxed = max(0.5 * rho, 1e-9)
            for idx in order:
                if idx in selected_set:
                    continue
                d_min = float(np.min(np.abs(velocities[idx] - velocities[np.array(selected, dtype=int)])))
                if d_min >= relaxed:
                    selected.append(idx)
                    selected_set.add(idx)
                if len(selected) >= upper_target:
                    break

        selected_points = [candidates[i] for i in selected]
        selected_points.sort(key=lambda p: p[2])

        self._last_main_cluster_ratio = float(main_ratio)
        self._last_selected_size = len(selected_points)
        return selected_points

    def _refresh_cache_if_needed(self) -> None:
        if not self._dirty:
            return
        self._cached_training_set = self._compute_merged_points()
        self.data = list(self._cached_training_set)
        self._dirty = False

    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        self._refresh_cache_if_needed()
        if not self._cached_training_set:
            return np.empty((0, 1)), np.empty((0, 1))
        arr = np.array(self._cached_training_set, dtype=float)
        return arr[:, 0:1], arr[:, 1:2]

    def get_training_set(self) -> List[Tuple[float, float]]:
        self._refresh_cache_if_needed()
        return [(p[0], p[1]) for p in self._cached_training_set]

    def get_diagnostics(self) -> dict:
        self._refresh_cache_if_needed()
        return {
            "main_cluster_ratio": float(self._last_main_cluster_ratio),
            "selected_size": int(self._last_selected_size),
        }

    def reset(self):
        for level in self.levels:
            level.reset()
        self.data = []
        self.total_adds = 0
        self._dirty = True
        self._cached_training_set = []
