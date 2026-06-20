from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class DynamicalSystem(Protocol):
    """Protocol for open-loop dynamical systems.

    Shape conventions:
    - state: ``(state_dim,)``
    - action: ``(action_dim,)``
    - disturbance: ``(disturbance_dim,)``
    - rollout states: ``(horizon + 1, state_dim)``
    - rollout actions: ``(horizon, action_dim)``
    - rollout disturbances: ``(horizon, disturbance_dim)``
    """

    system_id: str
    state_dim: int
    action_dim: int
    disturbance_dim: int

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Return an initial state with shape ``(state_dim,)``."""

    def step(
        self,
        state: np.ndarray,
        action: np.ndarray,
        disturbance: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Return the repaired next state with shape ``(state_dim,)``."""

    def rollout(
        self,
        initial_state: np.ndarray,
        actions: np.ndarray,
        disturbances: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Return states with shape ``(horizon + 1, state_dim)``."""

    def invariant_residual(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        disturbances: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Return per-step invariant residuals with shape ``(horizon,)``."""

    def repair_amount(self, raw_states: np.ndarray, repaired_states: np.ndarray) -> float:
        """Return a nonnegative scalar repair magnitude."""


@dataclass(frozen=True)
class TrajectoryBatch:
    """Batch of open-loop trajectories.

    Shape conventions:
    - ``states``: ``(n_trajectories, horizon + 1, state_dim)``
    - ``actions``: ``(n_trajectories, horizon, action_dim)``
    - ``disturbances``: ``(n_trajectories, horizon, disturbance_dim)``
    - ``scenario_type``: one label per trajectory
    """

    states: np.ndarray
    actions: np.ndarray
    disturbances: np.ndarray
    scenario_type: list[str]
    split: str
    system_id: str

    def __post_init__(self) -> None:
        if self.states.ndim != 3:
            raise ValueError("states must have shape (n, horizon + 1, state_dim)")
        if self.actions.ndim != 3:
            raise ValueError("actions must have shape (n, horizon, action_dim)")
        if self.disturbances.ndim != 3:
            raise ValueError("disturbances must have shape (n, horizon, disturbance_dim)")
        n, state_horizon, _ = self.states.shape
        n_actions, action_horizon, _ = self.actions.shape
        n_disturbances, disturbance_horizon, _ = self.disturbances.shape
        if n_actions != n or n_disturbances != n:
            raise ValueError("states, actions, and disturbances must have the same batch size")
        if action_horizon != state_horizon - 1:
            raise ValueError("actions horizon must be one less than states horizon")
        if disturbance_horizon != action_horizon:
            raise ValueError("disturbances horizon must match actions horizon")
        if len(self.scenario_type) != n:
            raise ValueError("scenario_type must contain one label per trajectory")
        if not np.isfinite(self.states).all():
            raise ValueError("states contain non-finite values")
        if not np.isfinite(self.actions).all():
            raise ValueError("actions contain non-finite values")
        if not np.isfinite(self.disturbances).all():
            raise ValueError("disturbances contain non-finite values")

    @property
    def n_trajectories(self) -> int:
        return int(self.states.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.actions.shape[1])

