from __future__ import annotations

import itertools

import numpy as np


def disagreement_score(predictions: list[np.ndarray]) -> float:
    if len(predictions) < 2:
        return 0.0
    distances = []
    for left, right in itertools.combinations(predictions, 2):
        left = np.asarray(left, dtype=float)
        right = np.asarray(right, dtype=float)
        if left.shape != right.shape:
            raise ValueError("prediction shapes must match")
        distances.append(float(np.sqrt(np.mean((left - right) ** 2))))
    return float(np.mean(distances))

