from __future__ import annotations

import numpy as np
import pandas as pd


def risk_coverage_curve(
    errors: np.ndarray,
    risk_scores: np.ndarray,
    bad_threshold: float,
    coverages: list[float],
) -> pd.DataFrame:
    errors = np.asarray(errors, dtype=float)
    risk_scores = np.asarray(risk_scores, dtype=float)
    if errors.ndim != 1 or risk_scores.ndim != 1:
        raise ValueError("errors and risk_scores must be one-dimensional")
    if len(errors) != len(risk_scores):
        raise ValueError("errors and risk_scores must have the same length")
    if len(errors) == 0:
        raise ValueError("at least one scenario is required")
    if not np.isfinite(errors).all() or not np.isfinite(risk_scores).all():
        raise ValueError("errors and risk_scores must be finite")

    order = np.argsort(risk_scores, kind="mergesort")
    sorted_errors = errors[order]
    rows = []
    n = len(errors)
    for coverage in coverages:
        if not 0.0 < coverage <= 1.0:
            raise ValueError("coverage values must be in (0, 1]")
        accepted_count = int(np.ceil(coverage * n))
        accepted_count = min(max(accepted_count, 1), n)
        accepted_errors = sorted_errors[:accepted_count]
        rejected_errors = sorted_errors[accepted_count:]
        false_accept_count = int(np.sum(accepted_errors > bad_threshold))
        false_accept_rate = false_accept_count / accepted_count
        rows.append(
            {
                "coverage": float(coverage),
                "false_accept_rate": float(false_accept_rate),
                "accepted_count": accepted_count,
                "false_accept_count": false_accept_count,
                "mean_error_accepted": float(np.mean(accepted_errors)),
                "mean_error_rejected": float(np.mean(rejected_errors)) if len(rejected_errors) else 0.0,
                "threshold": float(bad_threshold),
            }
        )
    return pd.DataFrame(rows)

