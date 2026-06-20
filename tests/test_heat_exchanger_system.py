from __future__ import annotations

import numpy as np

from scs.systems.heat_exchanger import HeatExchangerSystem


def test_heat_exchanger_rollout_is_bounded_and_nonconstant() -> None:
    system = HeatExchangerSystem()
    initial = system.reset(seed=5)
    actions = np.full((20, 1), 1.2)
    disturbances = np.full((20, 2), [118.0, 18.0])
    states = system.rollout(initial, actions, disturbances, dt=0.1)

    assert states.shape == (21, 2)
    assert np.all(states[:, 0] >= system.hot_bounds[0])
    assert np.all(states[:, 0] <= system.hot_bounds[1])
    assert np.all(states[:, 1] >= system.cold_bounds[0])
    assert np.all(states[:, 1] <= system.cold_bounds[1])
    assert not np.allclose(states[0], states[-1])


def test_heat_exchanger_repair_amount_positive_for_out_of_bounds_states() -> None:
    system = HeatExchangerSystem()
    raw = np.array([[20.0, 120.0], [90.0, 30.0]])
    repaired = system.clip_trajectory(raw)
    assert system.repair_amount(raw, repaired) > 0.0

