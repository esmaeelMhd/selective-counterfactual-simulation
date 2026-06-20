from __future__ import annotations

from typing import Protocol

import numpy as np

from scs.systems.base import TrajectoryBatch


class SimulatorModel(Protocol):
    """Interface for learned or baseline open-loop simulators."""

    model_id: str

    def fit(self, train: TrajectoryBatch) -> None:
        """Fit the model on a training trajectory batch."""

    def predict_rollout(
        self,
        initial_state: np.ndarray,
        actions: np.ndarray,
        disturbances: np.ndarray,
    ) -> np.ndarray:
        """Return predicted states with shape ``(horizon + 1, state_dim)``."""

    def predict_rollout_samples(
        self,
        initial_state: np.ndarray,
        actions: np.ndarray,
        disturbances: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        """Return sampled rollouts with shape ``(n_samples, horizon + 1, state_dim)``."""


def flatten_supervised(batch: TrajectoryBatch) -> tuple[np.ndarray, np.ndarray]:
    states_t = batch.states[:, :-1, :]
    next_states = batch.states[:, 1:, :]
    features = np.concatenate([states_t, batch.actions, batch.disturbances], axis=-1)
    targets = next_states - states_t
    return features.reshape(-1, features.shape[-1]), targets.reshape(-1, targets.shape[-1])


def validate_rollout_inputs(
    initial_state: np.ndarray,
    actions: np.ndarray,
    disturbances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    initial_state = np.asarray(initial_state, dtype=float)
    actions = np.asarray(actions, dtype=float)
    disturbances = np.asarray(disturbances, dtype=float)
    if initial_state.ndim != 1:
        raise ValueError("initial_state must have shape (state_dim,)")
    if actions.ndim != 2:
        raise ValueError("actions must have shape (horizon, action_dim)")
    if disturbances.ndim != 2:
        raise ValueError("disturbances must have shape (horizon, disturbance_dim)")
    if len(actions) != len(disturbances):
        raise ValueError("actions and disturbances must have the same horizon")
    return initial_state, actions, disturbances

