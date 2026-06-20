from __future__ import annotations

import numpy as np

from scs.systems.base import DynamicalSystem


def invariant_residual_score(
    system: DynamicalSystem,
    states: np.ndarray,
    actions: np.ndarray,
    disturbances: np.ndarray,
    dt: float,
) -> float:
    residual = system.invariant_residual(states, actions, disturbances, dt)
    return float(np.mean(np.asarray(residual, dtype=float)))

