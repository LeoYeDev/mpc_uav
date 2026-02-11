
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
                 min_distance: float = 0.01):
        super().__init__(max_size)
        self.novelty_weight = max(float(novelty_weight), 0.0)
        self.recency_weight = max(float(recency_weight), 0.0)
        if self.novelty_weight == 0.0 and self.recency_weight == 0.0:
            self.novelty_weight = 1.0
        self.decay_rate = max(float(decay_rate), 1e-6)
        self.min_distance = min_distance  # Minimum distance to consider adding

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
        # Compute nearest neighbor distance for each point (excluding itself).
        min_dists = np.zeros(n)
        for i in range(n):
            dists = np.abs(velocities - velocities[i])
            dists[i] = np.inf  # Ignore self
            min_dists[i] = np.min(dists)
            
        # Novelty uses adaptive scale (median nearest-neighbor distance) so that
        # one outlier does not collapse all other novelty scores.
        distance_scale = max(float(np.median(min_dists)), float(self.min_distance), 1e-6)
        novelty_scores = 1.0 - np.exp(-min_dists / distance_scale)
            
        # --- Recency Score ---
        # Exponential decay over "sample steps" (adaptive to timestamp scale)
        recency_scores = self._compute_recency_scores(timestamps)
        
        # Combine
        total_scores = self.novelty_weight * novelty_scores + self.recency_weight * recency_scores
        
        return total_scores

    def insert(self, v_scalar: float, y_scalar: float, timestamp: float) -> None:
        """
        Insert new point. If full, remove point with lowest IVS score.
        Also skips insertion if point is too close to existing points (redundant).
        """
        # check redundancy
        if self.data:
            velocities = np.array([p[0] for p in self.data])
            dist_to_existing = np.min(np.abs(velocities - v_scalar))
            if dist_to_existing < self.min_distance:
                # Near-duplicate in velocity space: refresh with latest residual and time.
                idx = np.argmin(np.abs(velocities - v_scalar))
                self.data[idx] = (float(v_scalar), float(y_scalar), float(timestamp))
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
    Multi-level IVS buffer.

    Motivation:
    - Level-0 keeps dense recent samples for fast adaptation.
    - Higher levels sub-sample in time and prefer novelty, preserving long-term
      coverage on rarely seen velocity regions.
    - Final training set is merged and re-selected to total max_size.
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
    ):
        super().__init__(max_size)
        self.novelty_weight = max(float(novelty_weight), 0.0)
        self.recency_weight = max(float(recency_weight), 0.0)
        if self.novelty_weight == 0.0 and self.recency_weight == 0.0:
            self.recency_weight = 1.0
        self.decay_rate = max(float(decay_rate), 1e-6)
        self.min_distance = max(float(min_distance), 0.0)
        self.merge_min_distance = (
            max(float(merge_min_distance), 0.0)
            if merge_min_distance is not None
            else max(self.min_distance, 1e-3)
        )

        self.level_capacities = self._resolve_level_capacities(max_size, level_capacities)
        self.level_sparsity = self._resolve_level_sparsity(
            n_levels=len(self.level_capacities),
            level_sparsity=level_sparsity,
        )
        self.levels = self._build_levels()

        self._dirty = True
        self._cached_training_set: List[Tuple[float, float, float]] = []

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
            sparse = [2 ** i for i in range(n_levels)]

        if len(sparse) < n_levels:
            last = sparse[-1] if sparse else 1
            for _ in range(n_levels - len(sparse)):
                last = max(1, last * 2)
                sparse.append(last)
        return sparse[:n_levels]

    def _build_levels(self) -> List[InformationGainBuffer]:
        levels: List[InformationGainBuffer] = []
        for i, cap in enumerate(self.level_capacities):
            # Higher levels slightly emphasize novelty while keeping recency relevant.
            nov_w = min(1.0, self.novelty_weight + 0.08 * i)
            rec_w = max(0.0, self.recency_weight - 0.08 * i)
            if nov_w + rec_w <= 1e-12:
                rec_w = 1.0
            norm = nov_w + rec_w
            nov_w /= norm
            rec_w /= norm

            level_decay = max(self.decay_rate / (1.0 + 0.3 * i), 1e-6)
            level_min_distance = self.min_distance * (1.0 + 0.5 * i)
            levels.append(
                InformationGainBuffer(
                    max_size=cap,
                    novelty_weight=nov_w,
                    recency_weight=rec_w,
                    decay_rate=level_decay,
                    min_distance=level_min_distance,
                )
            )
        return levels

    def insert(self, v_scalar: float, y_scalar: float, timestamp: float) -> None:
        v = float(v_scalar)
        y = float(y_scalar)
        t = float(timestamp)

        # Level-0 always receives every sample.
        self.levels[0].insert(v, y, t)

        # Higher levels sub-sample in time to retain slower memory.
        for level_idx in range(1, len(self.levels)):
            stride = self.level_sparsity[level_idx]
            if self.total_adds % stride == 0:
                self.levels[level_idx].insert(v, y, t)

        self.total_adds += 1
        self._dirty = True

    def _compute_merged_points(self) -> List[Tuple[float, float, float]]:
        candidates: List[Tuple[float, float, float, int]] = []
        for level_idx, level in enumerate(self.levels):
            for v, y, t in level.data:
                candidates.append((float(v), float(y), float(t), level_idx))

        if not candidates:
            return []

        # Prefer recent points first, then lower level index (more recent memory).
        candidates.sort(key=lambda p: (-p[2], p[3]))

        deduped: List[Tuple[float, float, float]] = []
        for v, y, t, _ in candidates:
            if not deduped:
                deduped.append((v, y, t))
                continue
            d = min(abs(v - e[0]) for e in deduped)
            if d >= self.merge_min_distance:
                deduped.append((v, y, t))

        # Final selection constrained by global max_size.
        selector = InformationGainBuffer(
            max_size=self.max_size,
            novelty_weight=self.novelty_weight,
            recency_weight=self.recency_weight,
            decay_rate=self.decay_rate,
            min_distance=self.merge_min_distance,
        )
        for v, y, t in sorted(deduped, key=lambda p: p[2]):
            selector.insert(v, y, t)
        return selector.data

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

    def reset(self):
        for level in self.levels:
            level.reset()
        self.data = []
        self.total_adds = 0
        self._dirty = True
        self._cached_training_set = []
