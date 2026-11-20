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


def validate_history(history: np.ndarray | None, batch_size: int) -> None:
    """Check an optional measured-history window of shape (batch, window, n_measured)."""
    if history is None:
        return
    history = np.asarray(history, dtype=float)
    if history.ndim != 3 or history.shape[0] != batch_size or history.shape[1] < 1 or history.shape[2] < 1:
        raise ValueError("history must have shape (batch, window, n_measured)")
    if not np.isfinite(history).all():
        raise ValueError("history contains non-finite values")


def rollout_from_delta(
    delta_batch,
    initial_states: np.ndarray,
    actions: np.ndarray,
    disturbances: np.ndarray,
    noise: np.ndarray | None = None,
    project=None,
) -> np.ndarray:
    """Recursive batch rollout.

    ``noise`` has shape (batch, horizon, state_dim). ``project(step, states)``
    may overwrite channels of each predicted state, such as flows that the
    plant sets from the influent, before that state is fed back.
    """
    n_rows, horizon, _ = actions.shape
    states = np.empty((n_rows, horizon + 1, initial_states.shape[-1]), dtype=float)
    states[:, 0] = initial_states
    for step in range(horizon):
        delta = np.asarray(delta_batch(states[:, step], actions[:, step], disturbances[:, step]), dtype=float)
        states[:, step + 1] = states[:, step] + delta
        if noise is not None:
            states[:, step + 1] = states[:, step + 1] + noise[:, step]
        if project is not None:
            states[:, step + 1] = project(step, states[:, step + 1])
    return states



def subset_batch(batch: TrajectoryBatch, rows: slice) -> TrajectoryBatch:
    """The trajectories in ``rows``, with the same split and system labels."""
    history = None if batch.measured_history is None else batch.measured_history[rows]
    return TrajectoryBatch(
        states=batch.states[rows],
        actions=batch.actions[rows],
        disturbances=batch.disturbances[rows],
        scenario_type=list(batch.scenario_type[rows]),
        split=batch.split,
        system_id=batch.system_id,
        measured_history=history,
    )


def time_blocked_split(n_rows: int, fraction: float, gap: int) -> tuple[slice, slice]:
    """Fit rows and validation rows from one training batch, in trajectory order.

    The validation rows are the last ``fraction`` of the trajectories. ``gap``
    trajectories before them are dropped, so that no fit window overlaps a
    validation window in time. This assumes trajectories are stored in time
    order, as the BSM1 generator writes them.
    """
    if not 0.0 < fraction < 1.0 or gap < 0:
        raise ValueError("fraction must be in (0, 1) and gap must not be negative")
    n_validation = max(1, int(round(n_rows * fraction)))
    stop = n_rows - n_validation - gap
    if stop < 1:
        raise ValueError(f"{n_rows} trajectories are too few for a validation block of {n_validation} and a gap of {gap}")
    return slice(0, stop), slice(n_rows - n_validation, n_rows)


def validation_rmse(model, batch: TrajectoryBatch, projection_factory=None) -> float:
    """Mean per-scenario RMSE of free-running rollouts on ``batch``.

    ``projection_factory(disturbances)`` returns the ``project`` hook for those
    disturbances, such as the BSM1 flow assignment. Without it nothing is projected.
    """
    project = None if projection_factory is None else projection_factory(batch.disturbances)
    predicted = model.predict_rollouts(batch.states[:, 0, :], batch.actions, batch.disturbances, project=project)
    return float(np.sqrt(np.mean((predicted - batch.states) ** 2, axis=(1, 2))).mean())


class EarlyStopping:
    """Keep the best snapshot and stop after ``patience`` checks without improvement."""

    def __init__(self, patience: int) -> None:
        if patience < 1:
            raise ValueError("patience must be at least 1")
        self.patience = int(patience)
        self.best_score = float("inf")
        self.best_epoch = 0
        self.best_state = None
        self.misses = 0
        self.curve: list[tuple[int, float]] = []

    def update(self, epoch: int, score: float, snapshot) -> bool:
        """Record a check at ``epoch``. Returns True when training should stop."""
        self.curve.append((int(epoch), float(score)))
        if score < self.best_score:
            self.best_score = float(score)
            self.best_epoch = int(epoch)
            self.best_state = snapshot()
            self.misses = 0
        else:
            self.misses += 1
        return self.misses >= self.patience
