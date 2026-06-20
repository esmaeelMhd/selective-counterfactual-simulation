from __future__ import annotations

import numpy as np

from scs.systems.base import DynamicalSystem


def repair_amount_score(system: DynamicalSystem, states: np.ndarray) -> float:
    raw_states = np.asarray(states, dtype=float)
    if hasattr(system, "clip_trajectory"):
        repaired_states = system.clip_trajectory(raw_states)  # type: ignore[attr-defined]
    else:
        repaired_states = raw_states
    return float(system.repair_amount(raw_states, repaired_states))

