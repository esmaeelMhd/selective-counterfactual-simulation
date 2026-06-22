from __future__ import annotations

from typing import Protocol

import numpy as np

from scs.models.base import validate_rollout_inputs
from scs.systems.base import TrajectoryBatch


class UserSimulatorModelProtocol(Protocol):
    """Protocol for user-supplied simulator models.

    Implementations intentionally match the existing ``SimulatorModel`` shape
    contract so custom models can be compared locally without changing the
    benchmark internals.
    """

    model_id: str

    def fit(self, train_batch: TrajectoryBatch) -> None:
        """Fit on a training trajectory batch."""

    def predict_rollout(
        self,
        initial_state: np.ndarray,
        actions: np.ndarray,
        disturbances: np.ndarray,
    ) -> np.ndarray:
        """Return predicted states with shape ``(horizon + 1, state_dim)``."""


class UserSimulatorModel:
    """Base class with validation helpers for bring-your-own models."""

    model_id = "user_model"

    def fit(self, train_batch: TrajectoryBatch) -> None:
        raise NotImplementedError("custom models must implement fit(train_batch)")

    def predict_rollout(
        self,
        initial_state: np.ndarray,
        actions: np.ndarray,
        disturbances: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError("custom models must implement predict_rollout")

    def predict_rollout_samples(
        self,
        initial_state: np.ndarray,
        actions: np.ndarray,
        disturbances: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        rollout = self.predict_rollout(initial_state, actions, disturbances)
        return np.repeat(rollout[None, :, :], int(n_samples), axis=0)

    @staticmethod
    def validate_inputs(
        initial_state: np.ndarray,
        actions: np.ndarray,
        disturbances: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return validate_rollout_inputs(initial_state, actions, disturbances)

    @staticmethod
    def validate_rollout_output(
        rollout: np.ndarray,
        initial_state: np.ndarray,
        actions: np.ndarray,
    ) -> np.ndarray:
        rollout = np.asarray(rollout, dtype=float)
        expected = (len(actions) + 1, len(initial_state))
        if rollout.shape != expected:
            raise ValueError(f"custom model rollout must have shape {expected}, got {rollout.shape}")
        if not np.isfinite(rollout).all():
            raise ValueError("custom model rollout contains non-finite values")
        return rollout
