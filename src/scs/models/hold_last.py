from __future__ import annotations

import numpy as np

from scs.models.base import validate_rollout_inputs
from scs.systems.base import TrajectoryBatch


class HoldLastModel:
    model_id = "hold_last"

    def __init__(self) -> None:
        self.state_dim: int | None = None

    def fit(self, train: TrajectoryBatch) -> None:
        self.state_dim = int(train.states.shape[-1])

    def predict_rollout(
        self,
        initial_state: np.ndarray,
        actions: np.ndarray,
        disturbances: np.ndarray,
    ) -> np.ndarray:
        initial_state, actions, disturbances = validate_rollout_inputs(initial_state, actions, disturbances)
        return np.repeat(initial_state[None, :], len(actions) + 1, axis=0)

    def predict_rollout_samples(
        self,
        initial_state: np.ndarray,
        actions: np.ndarray,
        disturbances: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        rollout = self.predict_rollout(initial_state, actions, disturbances)
        return np.repeat(rollout[None, :, :], n_samples, axis=0)

