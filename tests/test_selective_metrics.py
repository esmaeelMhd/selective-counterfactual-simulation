from __future__ import annotations

import numpy as np

from scs.metrics.selective import risk_coverage_curve


def test_risk_coverage_exact_known_case() -> None:
    errors = np.array([0.1, 0.4, 0.2, 0.8])
    risk = np.array([0.1, 0.2, 0.9, 0.3])
    curve = risk_coverage_curve(errors, risk, bad_threshold=0.3, coverages=[0.25, 0.5, 1.0])

    assert curve.loc[0, "accepted_count"] == 1
    assert curve.loc[0, "false_accept_count"] == 0
    assert curve.loc[0, "false_accept_rate"] == 0.0

    assert curve.loc[1, "accepted_count"] == 2
    assert curve.loc[1, "false_accept_count"] == 1
    assert curve.loc[1, "false_accept_rate"] == 0.5

    assert curve.loc[2, "accepted_count"] == 4
    assert curve.loc[2, "false_accept_count"] == 2
    assert curve.loc[2, "false_accept_rate"] == 0.5


def test_risk_coverage_rejects_bad_inputs() -> None:
    errors = np.array([0.1, np.nan])
    risk = np.array([0.1, 0.2])
    try:
        risk_coverage_curve(errors, risk, bad_threshold=0.3, coverages=[0.5])
    except ValueError:
        return
    raise AssertionError("expected ValueError for NaN errors")

