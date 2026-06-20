from __future__ import annotations

import numpy as np

from scs.models.base import SimulatorModel


def uncertainty_score(
    model: SimulatorModel,
    initial_state: np.ndarray,
    actions: np.ndarray,
    disturbances: np.ndarray,
    n_samples: int = 8,
) -> float:
    samples = model.predict_rollout_samples(initial_state, actions, disturbances, n_samples=n_samples)
    if samples.shape[0] != n_samples:
        raise ValueError("sample rollout count mismatch")
    return float(np.mean(np.std(samples, axis=0)))

