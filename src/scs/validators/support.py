from __future__ import annotations

import numpy as np

from scs.systems.base import TrajectoryBatch


class SupportDistance:
    """Standardized Euclidean distance from the training action/disturbance support."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, train: TrajectoryBatch) -> None:
        features = np.concatenate([train.actions, train.disturbances], axis=-1)
        flat = features.reshape(-1, features.shape[-1])
        self.mean_ = np.mean(flat, axis=0)
        self.std_ = np.maximum(np.std(flat, axis=0), 1e-6)

    def score(self, actions: np.ndarray, disturbances: np.ndarray) -> float:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("support distance must be fit before scoring")
        features = np.concatenate([actions, disturbances], axis=-1)
        z = (features - self.mean_) / self.std_
        return float(np.mean(np.linalg.norm(z, axis=-1)))

    def score_batch(self, batch: TrajectoryBatch) -> np.ndarray:
        return np.asarray(
            [self.score(batch.actions[i], batch.disturbances[i]) for i in range(batch.n_trajectories)],
            dtype=float,
        )

