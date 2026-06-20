from __future__ import annotations

import numpy as np

from scs.systems.two_tank import TwoTankSystem


def test_step_and_rollout_shapes() -> None:
    system = TwoTankSystem()
    state = np.array([4.0, 3.0])
    action = np.array([0.8])
    disturbance = np.array([0.5, 0.45])
    next_state = system.step(state, action, disturbance, dt=0.1)
    assert next_state.shape == (2,)

    actions = np.full((12, 1), 0.8)
    disturbances = np.full((12, 2), [0.5, 0.45])
    states = system.rollout(state, actions, disturbances, dt=0.1)
    assert states.shape == (13, 2)
    assert np.all(states >= 0.0)
    assert not np.allclose(states[0], states[-1])


def test_repair_amount_positive_for_invalid_state() -> None:
    system = TwoTankSystem()
    raw = np.array([[-1.0, 12.0], [2.0, 3.0]])
    repaired = system.clip_trajectory(raw)
    assert system.repair_amount(raw, repaired) > 0.0
    assert np.all(repaired >= 0.0)
    assert np.all(repaired <= system.capacity)


def test_closed_system_conservation_residual_near_zero_without_clipping() -> None:
    system = TwoTankSystem()
    initial = np.array([5.0, 2.5])
    actions = np.full((20, 1), 0.5)
    disturbances = np.zeros((20, 2))
    states = system.rollout(initial, actions, disturbances, dt=0.05)
    residual = system.invariant_residual(states, actions, disturbances, dt=0.05)
    assert float(np.max(residual)) < 1e-10

