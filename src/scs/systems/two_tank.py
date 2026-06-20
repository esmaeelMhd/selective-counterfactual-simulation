from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TwoTankSystem:
    """Two-tank inventory system with transfer, inflow, demand, and clipping repair."""

    capacity: np.ndarray = field(default_factory=lambda: np.array([10.0, 10.0], dtype=float))
    pump_gain: float = 1.15
    eps: float = 1e-6

    system_id: str = "two_tank"
    state_dim: int = 2
    action_dim: int = 1
    disturbance_dim: int = 2

    def reset(self, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return np.array(
            [
                rng.uniform(3.6, 5.2),
                rng.uniform(2.1, 3.9),
            ],
            dtype=float,
        )

    def transfer_flow(self, state: np.ndarray, action: np.ndarray) -> float:
        state = np.asarray(state, dtype=float)
        action = np.asarray(action, dtype=float)
        command = float(np.clip(action[0], 0.0, 2.5))
        head = max(float(state[0] - state[1]), 0.0)
        return self.pump_gain * command * float(np.sqrt(head + self.eps))

    def raw_step(
        self,
        state: np.ndarray,
        action: np.ndarray,
        disturbance: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        state = np.asarray(state, dtype=float)
        disturbance = np.asarray(disturbance, dtype=float)
        q12 = self.transfer_flow(state, action)
        d_in = float(disturbance[0])
        d_out = float(disturbance[1])
        dx = np.array([d_in - q12, q12 - d_out], dtype=float)
        return state + dt * dx

    def repair_state(self, raw_state: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(raw_state, dtype=float), 0.0, self.capacity)

    def step(
        self,
        state: np.ndarray,
        action: np.ndarray,
        disturbance: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        return self.repair_state(self.raw_step(state, action, disturbance, dt))

    def rollout(
        self,
        initial_state: np.ndarray,
        actions: np.ndarray,
        disturbances: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        actions = np.asarray(actions, dtype=float)
        disturbances = np.asarray(disturbances, dtype=float)
        if actions.ndim != 2 or actions.shape[1] != self.action_dim:
            raise ValueError("actions must have shape (horizon, 1)")
        if disturbances.ndim != 2 or disturbances.shape[1] != self.disturbance_dim:
            raise ValueError("disturbances must have shape (horizon, 2)")
        if len(actions) != len(disturbances):
            raise ValueError("actions and disturbances must have the same horizon")

        states = np.empty((len(actions) + 1, self.state_dim), dtype=float)
        states[0] = self.repair_state(initial_state)
        for t in range(len(actions)):
            states[t + 1] = self.step(states[t], actions[t], disturbances[t], dt)
        return states

    def invariant_residual(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        disturbances: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        states = np.asarray(states, dtype=float)
        disturbances = np.asarray(disturbances, dtype=float)
        if states.ndim != 2:
            raise ValueError("states must have shape (horizon + 1, state_dim)")
        inventory_change = states[1:].sum(axis=1) - states[:-1].sum(axis=1)
        external_balance = dt * (disturbances[:, 0] - disturbances[:, 1])
        return np.abs(inventory_change - external_balance)

    def repair_amount(self, raw_states: np.ndarray, repaired_states: np.ndarray) -> float:
        raw_states = np.asarray(raw_states, dtype=float)
        repaired_states = np.asarray(repaired_states, dtype=float)
        return float(np.mean(np.abs(raw_states - repaired_states)))

    def clip_trajectory(self, states: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(states, dtype=float), 0.0, self.capacity)


def make_two_tank_system() -> TwoTankSystem:
    return TwoTankSystem()

