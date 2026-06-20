from __future__ import annotations

import numpy as np


def _diff(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    if y_pred.shape != y_true.shape:
        raise ValueError(f"shape mismatch: {y_pred.shape} != {y_true.shape}")
    return y_pred - y_true


def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean(_diff(y_pred, y_true) ** 2))


def mae(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean(np.abs(_diff(y_pred, y_true))))


def rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.sqrt(mse(y_pred, y_true)))


def max_abs_error(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.max(np.abs(_diff(y_pred, y_true))))


def final_state_error(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(y_pred)[-1] - np.asarray(y_true)[-1]))

