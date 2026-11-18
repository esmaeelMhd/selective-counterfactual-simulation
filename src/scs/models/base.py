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
        history: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return predicted states with shape ``(horizon + 1, state_dim)``.

        ``history`` is an optional window of measured variables. Existing
        models ignore it, so the three-argument call is unchanged.
        """

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


def validate_rollout_batch(
    initial_states: np.ndarray,
    actions: np.ndarray,
    disturbances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    initial_states = np.asarray(initial_states, dtype=float)
    actions = np.asarray(actions, dtype=float)
    disturbances = np.asarray(disturbances, dtype=float)
    if initial_states.ndim != 2:
        raise ValueError("initial_states must have shape (batch, state_dim)")
    if actions.ndim != 3 or disturbances.ndim != 3:
        raise ValueError("actions and disturbances must have shape (batch, horizon, dim)")
    if initial_states.shape[0] != actions.shape[0] or actions.shape[:2] != disturbances.shape[:2]:
        raise ValueError("batch rollout inputs must share batch and horizon")
    return initial_states, actions, disturbances


def rollout_from_delta(
    delta_batch,
    initial_states: np.ndarray,
    actions: np.ndarray,
    disturbances: np.ndarray,
    noise: np.ndarray | None = None,
) -> np.ndarray:
    """Recursive batch rollout. ``noise`` has shape (batch, horizon, state_dim)."""
    n_rows, horizon, _ = actions.shape
    states = np.empty((n_rows, horizon + 1, initial_states.shape[-1]), dtype=float)
    states[:, 0] = initial_states
    for step in range(horizon):
        delta = np.asarray(delta_batch(states[:, step], actions[:, step], disturbances[:, step]), dtype=float)
        states[:, step + 1] = states[:, step] + delta
        if noise is not None:
            states[:, step + 1] = states[:, step + 1] + noise[:, step]
    return states

