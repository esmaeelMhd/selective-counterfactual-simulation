from __future__ import annotations

import numpy as np

from scs.systems.cstr import CSTRSystem


def test_cstr_rollout_is_bounded_and_nonconstant() -> None:
    system = CSTRSystem()
    initial = system.reset(seed=3)
    actions = np.full((18, 1), 9.5)
    disturbances = np.full((18, 1), 1.0)
    states = system.rollout(initial, actions, disturbances, dt=0.1)

    assert states.shape == (19, 2)
    assert np.all(states[:, 0] >= system.concentration_bounds[0])
    assert np.all(states[:, 0] <= system.concentration_bounds[1])
    assert np.all(states[:, 1] >= system.temperature_bounds[0])
    assert np.all(states[:, 1] <= system.temperature_bounds[1])
    assert not np.allclose(states[0], states[-1])


def test_cstr_repair_amount_positive_for_out_of_bounds_states() -> None:
    system = CSTRSystem()
    raw = np.array([[-0.2, 600.0], [1.0, 340.0]])
    repaired = system.clip_trajectory(raw)
    assert system.repair_amount(raw, repaired) > 0.0

