from __future__ import annotations

import numpy as np

from scs.metrics.events import concentration_below_limit, constraint_violation_event, temperature_above_limit


def test_cstr_event_metrics_detect_temperature_and_concentration_limits() -> None:
    states = np.array(
        [
            [1.0, 350.0],
            [0.2, 360.0],
            [0.8, 410.0],
        ],
        dtype=float,
    )

    assert temperature_above_limit(states, limit=390.0).tolist() == [False, False, True]
    assert concentration_below_limit(states, limit=0.25).tolist() == [False, True, False]
    assert constraint_violation_event(states, temperature_limit=390.0, concentration_limit=0.25).tolist() == [
        False,
        True,
        True,
    ]
