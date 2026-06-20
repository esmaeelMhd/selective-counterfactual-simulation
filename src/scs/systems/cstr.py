from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CSTRSystem:
    """Small CSTR-style reaction system with bounded concentration and temperature."""

    system_id: str = "cstr"
    state_dim: int = 2
    action_dim: int = 1
    disturbance_dim: int = 3
    concentration_bounds: tuple[float, float] = (0.0, 2.0)
    temperature_bounds: tuple[float, float] = (250.0, 500.0)

    def reset(self, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return np.array([rng.uniform(0.8, 1.2), rng.uniform(330.0, 360.0)], dtype=float)

    def raw_step(
        self,
        state: np.ndarray,
        action: np.ndarray,
        disturbance: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        concentration, temperature = np.asarray(state, dtype=float)
        cooling = float(action[0])
        feed_concentration = float(disturbance[0])
        feed_temperature = float(disturbance[1])
        flow_rate = float(disturbance[2])
        reaction_rate = 0.18 * concentration * np.exp((temperature - 330.0) / 80.0)
        d_concentration = flow_rate * (feed_concentration - concentration) - reaction_rate
        heat_release = 0.8 * reaction_rate * 40.0
        heat_removal = 0.45 * cooling
        heat_exchange = 0.08 * (feed_temperature - temperature)
        d_temperature = flow_rate * heat_exchange + heat_release - heat_removal
        return np.array(
            [
                concentration + dt * d_concentration,
                temperature + dt * d_temperature,
            ],
            dtype=float,
        )

    def repair_state(self, raw_state: np.ndarray) -> np.ndarray:
        raw_state = np.asarray(raw_state, dtype=float)
        return np.array(
            [
                np.clip(raw_state[0], *self.concentration_bounds),
                np.clip(raw_state[1], *self.temperature_bounds),
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
        repaired[..., 0] = np.clip(repaired[..., 0], *self.concentration_bounds)
        repaired[..., 1] = np.clip(repaired[..., 1], *self.temperature_bounds)
        return repaired
