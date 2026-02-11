
"""
Data buffer strategies for Online GP.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple

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
               Normalized to [0, 1] relative to the max distance in current buffer.
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
            
        # Normalize novelty to [0, 1]
        max_dist = np.max(min_dists)
        if max_dist > 1e-9:
            novelty_scores = min_dists / max_dist
        else:
            novelty_scores = np.zeros(n) # All points are identical
            
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
            if dist_to_existing < self.min_distance and len(self.data) >= self.max_size // 2:
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
