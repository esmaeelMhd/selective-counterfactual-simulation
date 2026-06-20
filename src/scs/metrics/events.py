from __future__ import annotations

import numpy as np


def overflow_event(states: np.ndarray, capacity: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=float)
    capacity = np.asarray(capacity, dtype=float)
    return np.any(states > capacity, axis=-1)


def underflow_event(states: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=float)
    return np.any(states < 0.0, axis=-1)


def threshold_crossing_event(states: np.ndarray, threshold: float, state_index: int = 1) -> np.ndarray:
    states = np.asarray(states, dtype=float)
    return states[..., state_index] >= threshold


def event_precision_recall(predicted: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    predicted = np.asarray(predicted, dtype=bool)
    actual = np.asarray(actual, dtype=bool)
    tp = float(np.sum(predicted & actual))
    fp = float(np.sum(predicted & ~actual))
    fn = float(np.sum(~predicted & actual))
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    return {"precision": precision, "recall": recall}

