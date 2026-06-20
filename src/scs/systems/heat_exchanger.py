from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HeatExchangerSystem:
    """Two-state heat exchanger with hot and cold outlet temperatures.

    State:
    - ``x[0]`` hot-side outlet temperature
    - ``x[1]`` cold-side outlet temperature

    Action:
    - ``u[0]`` coolant-flow command

    Disturbance:
    - ``d[0]`` hot inlet temperature
    - ``d[1]`` cold inlet temperature
    """

    system_id: str = "heat_exchanger"
    state_dim: int = 2
    action_dim: int = 1
    disturbance_dim: int = 2
    hot_bounds: tuple[float, float] = (40.0, 140.0)
    cold_bounds: tuple[float, float] = (5.0, 95.0)
    heat_transfer_gain: float = 0.018

    def reset(self, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return np.array([rng.uniform(86.0, 96.0), rng.uniform(24.0, 32.0)], dtype=float)

    def raw_step(
        self,
        state: np.ndarray,
        action: np.ndarray,
        disturbance: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        hot_out, cold_out = np.asarray(state, dtype=float)
        coolant_flow = float(np.clip(action[0], 0.1, 4.0))
        hot_in, cold_in = np.asarray(disturbance, dtype=float)
        approach = max(hot_out - cold_out, 0.0)
        transfer = self.heat_transfer_gain * coolant_flow * approach
        d_hot = 0.34 * (hot_in - hot_out) - transfer
        d_cold = 0.28 * (cold_in - cold_out) + 0.82 * transfer
        return np.array([hot_out + dt * d_hot, cold_out + dt * d_cold], dtype=float)

    def repair_state(self, raw_state: np.ndarray) -> np.ndarray:
        raw_state = np.asarray(raw_state, dtype=float)
        return np.array(
            [
                np.clip(raw_state[0], *self.hot_bounds),
                np.clip(raw_state[1], *self.cold_bounds),
            ],
            dtype=float,
        )

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
        predicted = []
        for t in range(len(actions)):
            predicted.append(self.raw_step(states[t], actions[t], disturbances[t], dt))
        return np.linalg.norm(np.asarray(predicted) - states[1:], axis=1)

    def repair_amount(self, raw_states: np.ndarray, repaired_states: np.ndarray) -> float:
        return float(np.mean(np.abs(np.asarray(raw_states) - np.asarray(repaired_states))))

    def clip_trajectory(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=float)
        repaired = states.copy()
        repaired[..., 0] = np.clip(repaired[..., 0], *self.hot_bounds)
        repaired[..., 1] = np.clip(repaired[..., 1], *self.cold_bounds)
        return repaired

