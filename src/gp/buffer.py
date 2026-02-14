"""
Data buffer strategies for Online GP.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


class BaseBuffer(ABC):
    """Abstract base class for data buffers."""

    def __init__(self, max_size: int):
        self.max_size = int(max(1, max_size))
        self.data: List[Tuple[float, float, float]] = []  # (v, y, t)
        self.total_adds = 0

    @abstractmethod
    def insert(self, v_scalar: float, y_scalar: float, timestamp: float) -> None:
        """Insert a new data point into the buffer."""

    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return current buffer data as numpy arrays (X, Y)."""
        if not self.data:
            return np.empty((0, 1)), np.empty((0, 1))
        arr = np.asarray(self.data, dtype=float)
        return arr[:, 0:1], arr[:, 1:2]

    def get_training_set(self) -> List[Tuple[float, float]]:
        """Return list of (v, y) tuples for training."""
        return [(p[0], p[1]) for p in self.data]

    def get_training_set_full(self) -> List[Tuple[float, float]]:
        """Return full training set (default same as get_training_set)."""
        return self.get_training_set()

    def get_effective_size_fast(self) -> int:
        return int(len(self.data))

    def get_velocity_bounds_fast(self) -> Tuple[Optional[float], Optional[float]]:
        if not self.data:
            return None, None
        v = np.asarray([p[0] for p in self.data], dtype=float)
        return float(np.min(v)), float(np.max(v))

    def get_diagnostics_fast(self) -> Dict[str, float]:
        size = float(len(self.data))
        return {
            "selected_size": size,
            "unique_ratio": 1.0 if size > 0 else 0.0,
            "duplicate_ratio": 0.0,
            "no_delete_phase": 0.0,
            "insert_accept_ratio": 1.0 if self.total_adds > 0 else 0.0,
            "insert_skip_ratio": 0.0,
            "prune_old_count": 0.0,
            "flip_delete_count": 0.0,
            "prune_old_count_last": 0.0,
            "flip_delete_count_last": 0.0,
        }

    def get_diagnostics(self) -> Dict[str, float]:
        return self.get_diagnostics_fast()

    def get_full_merge_call_count(self) -> int:
        return 0

    def reset(self):
        self.data = []
        self.total_adds = 0


class FIFOBuffer(BaseBuffer):
    """First-In-First-Out Buffer."""

    def insert(self, v_scalar: float, y_scalar: float, timestamp: float) -> None:
        if len(self.data) >= self.max_size:
            self.data.pop(0)
        self.data.append((float(v_scalar), float(y_scalar), float(timestamp)))
        self.total_adds += 1


@dataclass
class _LevelStore:
    """Simple level storage with fixed capacity."""

    capacity: int
    data: List[Tuple[float, float, float]] = field(default_factory=list)


class MultiLevelInformationGainBuffer(BaseBuffer):
    """
    Simplified multi-level IVS buffer (rule-driven):
    1) Keep 3 levels (L0/L1/L2) with fixed sparsity [1, 2, 5].
    2) New point gating by 2D Euclidean distance to newest L0 point: (v, y).
    3) After insertion, prune all older near points by x-axis distance only: |Δv|.
    4) Direction-flip cleanup removes oldest point iteratively (bounded by limit).
    5) Final training set is merged chronologically and capped at N.
    """

    def __init__(
        self,
        max_size: int,
        insert_min_delta_v: float = 0.15,
        prune_old_delta_v: float = 0.15,
        flip_prune_limit: int = 3,
        no_prune_below_n: int = 5,
        level_capacities: Optional[List[int]] = None,
        level_sparsity: Optional[List[int]] = None,
        novelty_weight: float = 0.55,
        recency_weight: float = 0.45,
        recency_decay_rate: float = 0.10,
    ):
        super().__init__(max_size)
        self.insert_min_delta_v = max(float(insert_min_delta_v), 0.0)
        self.prune_old_delta_v = max(float(prune_old_delta_v), 0.0)
        self.flip_prune_limit = max(0, int(flip_prune_limit))
        # 保护上限，避免阈值设置超过总容量导致长期不收敛。
        self.no_prune_below_n = max(1, min(int(no_prune_below_n), self.max_size))
        self.novelty_weight = max(0.0, float(novelty_weight))
        self.recency_weight = max(0.0, float(recency_weight))
        self.recency_decay_rate = max(0.0, float(recency_decay_rate))

        # 三层容量：优先使用手动配置；若无效则回退自动分配（60/25/15）。
        self.level_capacities = self._resolve_level_capacities(self.max_size, level_capacities)
        if level_sparsity is None:
            self.level_sparsity = [1, 2, 5]
        else:
            sparse = [max(1, int(v)) for v in level_sparsity[:3]]
            while len(sparse) < 3:
                sparse.append([1, 2, 5][len(sparse)])
            self.level_sparsity = sparse

        self.levels: List[_LevelStore] = [_LevelStore(capacity=c) for c in self.level_capacities]

        self._accepted_adds = 0
        self.insert_accept_count = 0
        self.insert_skip_count = 0
        self.prune_old_count_total = 0
        self.flip_delete_count_total = 0
        self._last_prune_old_count = 0
        self._last_flip_delete_count = 0

        self._cached_training_set: List[Tuple[float, float, float]] = []
        self._dirty = True
        self._merge_refresh_calls = 0

        self._last_selected_size = 0
        self._last_candidate_count = 0
        self._last_unique_candidate_count = 0
        self._last_unique_ratio = 0.0
        self._last_duplicate_ratio = 0.0
        self._last_no_delete_phase = 1.0

    def _resolve_level_capacities(self, max_size: int, manual_caps: Optional[List[int]] = None) -> List[int]:
        n = int(max(1, max_size))
        if manual_caps is not None and len(manual_caps) > 0:
            caps = [max(0, int(v)) for v in list(manual_caps[:3])]
            while len(caps) < 3:
                caps.append(0)
            total = int(sum(caps))
            if total <= 0:
                # fallback to auto if manual is invalid
                return self._resolve_level_capacities(n, None)
            # 将手动容量归一化到总和=n，保持比例并避免异常输入导致不公平比较。
            scaled = np.asarray(caps, dtype=float) * (float(n) / float(total))
            caps = np.floor(scaled).astype(int)
            remain = int(n - int(np.sum(caps)))
            frac_order = list(np.argsort(-(scaled - caps)))
            idx = 0
            while remain > 0:
                caps[int(frac_order[idx % 3])] += 1
                remain -= 1
                idx += 1
            out = [int(caps[0]), int(caps[1]), int(caps[2])]
            out[0] += int(n - sum(out))
            return out

        ratios = np.asarray([0.60, 0.25, 0.15], dtype=float)
        raw = ratios * float(n)
        caps = np.floor(raw).astype(int)
        remain = int(n - int(np.sum(caps)))

        # distribute remainders by largest fractional parts
        frac_order = list(np.argsort(-(raw - caps)))
        idx = 0
        while remain > 0:
            caps[int(frac_order[idx % 3])] += 1
            remain -= 1
            idx += 1

        # avoid negative and keep exact sum
        caps = np.maximum(0, caps)
        diff = int(n - int(np.sum(caps)))
        if diff != 0:
            caps[0] += diff

        out = [int(caps[0]), int(caps[1]), int(caps[2])]
        # safety: exact sum equals n
        out[0] += int(n - sum(out))
        return out

    @staticmethod
    def _point_key(p: Tuple[float, float, float]) -> Tuple[float, float, float]:
        return (round(float(p[0]), 9), round(float(p[1]), 9), round(float(p[2]), 9))

    def _compute_score(self, points: List[Tuple[float, float, float]], idx: int, t_ref: float) -> float:
        """统一信息价值分数：新颖性 + 时效性。"""
        if not points:
            return 0.0

        v_i, _, t_i = points[idx]
        # 时效性：越新越高
        age = max(0.0, float(t_ref) - float(t_i))
        recency = float(np.exp(-self.recency_decay_rate * age))

        # 新颖性：最近邻越远越高
        if len(points) <= 1:
            novelty = 1.0
        else:
            dists = [abs(float(v_i) - float(p[0])) for j, p in enumerate(points) if j != idx]
            d_min = min(dists) if dists else 0.0
            v_all = np.asarray([p[0] for p in points], dtype=float)
            spread = float(np.std(v_all) + 1e-6)
            novelty = float(1.0 - np.exp(-d_min / spread))

        w_n = max(0.0, self.novelty_weight)
        w_r = max(0.0, self.recency_weight)
        w_sum = w_n + w_r
        if w_sum < 1e-12:
            return recency
        return float((w_n * novelty + w_r * recency) / w_sum)

    def _append_with_capacity(
        self,
        level_idx: int,
        point: Tuple[float, float, float],
        enforce_capacity: bool = True,
    ) -> None:
        level = self.levels[level_idx]
        if level.capacity <= 0:
            return
        level.data.append(point)
        if not enforce_capacity:
            return
        while len(level.data) > level.capacity:
            t_ref = max(float(p[2]) for p in level.data)
            scores = [self._compute_score(level.data, i, t_ref=t_ref) for i in range(len(level.data))]
            remove_idx = int(np.argmin(scores))
            level.data.pop(remove_idx)

    def _collect_candidates(self) -> List[Tuple[float, float, float]]:
        candidates: List[Tuple[float, float, float]] = []
        for level in self.levels:
            candidates.extend(level.data)
        return candidates

    def _merge_unique_chronological(self) -> List[Tuple[float, float, float]]:
        candidates = self._collect_candidates()
        if not candidates:
            return []
        candidates = sorted(candidates, key=lambda p: p[2])
        seen = set()
        merged: List[Tuple[float, float, float]] = []
        for p in candidates:
            k = self._point_key(p)
            if k in seen:
                continue
            seen.add(k)
            merged.append((float(p[0]), float(p[1]), float(p[2])))
        if len(merged) > self.max_size:
            merged = merged[-self.max_size :]
        return merged

    def _prune_old_near_points(self, v_new: float, t_new: float) -> int:
        removed = 0
        threshold = float(self.prune_old_delta_v)
        for level in self.levels:
            kept: List[Tuple[float, float, float]] = []
            for p in level.data:
                is_older = float(p[2]) < float(t_new)
                is_near = abs(float(p[0]) - float(v_new)) < threshold
                if is_older and is_near:
                    removed += 1
                else:
                    kept.append(p)
            level.data = kept
        return int(removed)

    def _remove_oldest_point_from_all_levels(self) -> bool:
        merged = self._merge_unique_chronological()
        if not merged:
            return False
        oldest = merged[0]
        oldest_key = self._point_key(oldest)
        removed_any = False
        for level in self.levels:
            kept: List[Tuple[float, float, float]] = []
            for p in level.data:
                if self._point_key(p) == oldest_key:
                    removed_any = True
                    continue
                kept.append(p)
            level.data = kept
        return bool(removed_any)

    def _apply_flip_pruning(self) -> int:
        removed_steps = 0
        max_iter = int(self.flip_prune_limit)
        for _ in range(max_iter):
            merged = self._merge_unique_chronological()
            if len(merged) < 4:
                break

            old_delta = float(merged[1][0] - merged[0][0])
            new_delta = float(merged[-1][0] - merged[-2][0])
            if abs(old_delta) < 1e-12 or abs(new_delta) < 1e-12:
                break

            if np.sign(old_delta) * np.sign(new_delta) < 0.0:
                if self._remove_oldest_point_from_all_levels():
                    removed_steps += 1
                    continue
            break
        return int(removed_steps)

    def _refresh_cache_if_needed(self, force: bool = False) -> None:
        if not force and not self._dirty:
            return

        merged = self._merge_unique_chronological()
        candidate_count = int(sum(len(level.data) for level in self.levels))
        unique_count = int(len(merged))

        self._cached_training_set = merged
        self.data = list(merged)
        self._last_selected_size = unique_count
        self._last_candidate_count = candidate_count
        self._last_unique_candidate_count = unique_count
        if candidate_count > 0:
            self._last_unique_ratio = float(unique_count / candidate_count)
        else:
            self._last_unique_ratio = 0.0
        self._last_duplicate_ratio = float(max(0.0, 1.0 - self._last_unique_ratio))

        self._merge_refresh_calls += 1
        self._dirty = False

    def insert(self, v_scalar: float, y_scalar: float, timestamp: float) -> None:
        v_new = float(v_scalar)
        y_new = float(y_scalar)
        t_new = float(timestamp)
        self.total_adds += 1
        self._last_prune_old_count = 0
        self._last_flip_delete_count = 0
        # 当有效点数 < no_prune_below_n 时，禁用所有删点逻辑，优先积累样本。
        self._refresh_cache_if_needed()
        selected_size = int(self._last_selected_size)
        in_no_delete_phase = selected_size < int(self.no_prune_below_n)
        self._last_no_delete_phase = 1.0 if in_no_delete_phase else 0.0

        level0 = self.levels[0]
        accept = False
        if level0.capacity > 0:
            if len(level0.data) < level0.capacity:
                accept = True
            elif level0.data:
                latest = level0.data[-1]
                dv = float(v_new - float(latest[0]))
                dy = float(y_new - float(latest[1]))
                dist_2d = float(np.hypot(dv, dy))
                accept = bool(dist_2d >= self.insert_min_delta_v)
            else:
                accept = True

        if not accept:
            self.insert_skip_count += 1
            return

        self.insert_accept_count += 1
        self._accepted_adds += 1
        p_new = (v_new, y_new, t_new)

        # L0 每个接受点都进入。
        self._append_with_capacity(0, p_new, enforce_capacity=not in_no_delete_phase)

        # L1/L2 按步长稀疏提升。
        for level_idx in (1, 2):
            if self.levels[level_idx].capacity <= 0:
                continue
            stride = max(1, int(self.level_sparsity[level_idx]))
            if self._accepted_adds % stride == 0:
                self._append_with_capacity(level_idx, p_new, enforce_capacity=not in_no_delete_phase)

        if not in_no_delete_phase:
            pruned = self._prune_old_near_points(v_new=v_new, t_new=t_new)
            self.prune_old_count_total += int(pruned)
            self._last_prune_old_count = int(pruned)

            flip_removed = self._apply_flip_pruning()
            self.flip_delete_count_total += int(flip_removed)
            self._last_flip_delete_count = int(flip_removed)

        self._dirty = True

    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        self._refresh_cache_if_needed()
        return super().get_data()

    def get_training_set(self) -> List[Tuple[float, float]]:
        self._refresh_cache_if_needed()
        return [(p[0], p[1]) for p in self._cached_training_set]

    def get_training_set_full(self) -> List[Tuple[float, float]]:
        self._refresh_cache_if_needed(force=True)
        return [(p[0], p[1]) for p in self._cached_training_set]

    def get_effective_size_fast(self) -> int:
        self._refresh_cache_if_needed()
        return int(self._last_selected_size)

    def get_velocity_bounds_fast(self) -> Tuple[Optional[float], Optional[float]]:
        self._refresh_cache_if_needed()
        if not self._cached_training_set:
            return None, None
        v = np.asarray([p[0] for p in self._cached_training_set], dtype=float)
        return float(np.min(v)), float(np.max(v))

    def get_diagnostics_fast(self) -> Dict[str, float]:
        self._refresh_cache_if_needed()
        total_seen = float(max(1, self.total_adds))
        return {
            "selected_size": float(self._last_selected_size),
            "unique_ratio": float(self._last_unique_ratio),
            "duplicate_ratio": float(self._last_duplicate_ratio),
            "no_delete_phase": float(self._last_no_delete_phase),
            "insert_accept_ratio": float(self.insert_accept_count / total_seen),
            "insert_skip_ratio": float(self.insert_skip_count / total_seen),
            "prune_old_count": float(self.prune_old_count_total),
            "flip_delete_count": float(self.flip_delete_count_total),
            "prune_old_count_last": float(self._last_prune_old_count),
            "flip_delete_count_last": float(self._last_flip_delete_count),
        }

    def get_diagnostics(self) -> Dict[str, float]:
        diag = dict(self.get_diagnostics_fast())
        diag.update(
            {
                "level_capacities": [int(c) for c in self.level_capacities],
                "level_sparsity": [int(s) for s in self.level_sparsity],
            }
        )
        return diag

    def get_full_merge_call_count(self) -> int:
        return int(self._merge_refresh_calls)

    def reset(self):
        for level in self.levels:
            level.data = []
        self.data = []
        self.total_adds = 0
        self._accepted_adds = 0
        self.insert_accept_count = 0
        self.insert_skip_count = 0
        self.prune_old_count_total = 0
        self.flip_delete_count_total = 0
        self._last_prune_old_count = 0
        self._last_flip_delete_count = 0

        self._cached_training_set = []
        self._dirty = True
        self._merge_refresh_calls = 0

        self._last_selected_size = 0
        self._last_candidate_count = 0
        self._last_unique_candidate_count = 0
        self._last_unique_ratio = 0.0
        self._last_duplicate_ratio = 0.0
        self._last_no_delete_phase = 1.0
