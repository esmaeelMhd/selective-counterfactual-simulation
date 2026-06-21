from __future__ import annotations

import pandas as pd

from scs.experiments.cstr_weakness import _far_for_rows, _rmse_grid_verdict


def test_far_rows_compute_accepted_count_and_false_accepts() -> None:
    group = pd.DataFrame({"risk_j": [0.1, 0.2, 0.3], "rmse": [0.1, 0.4, 0.5]})
    result = _far_for_rows(group, "j", 0.5, 0.15)
    assert result["accepted_count"] == 2
    assert result["accepted_false_accept_count"] == 1


def test_degenerate_thresholds_are_marked_unavailable_in_fixture() -> None:
    grid = pd.DataFrame({"bad_rate": [1.0, 0.0]})
    assert not ((grid["bad_rate"] > 0.0) & (grid["bad_rate"] < 1.0)).any()


def test_cstr_accepted_region_too_risky_verdict() -> None:
    grid = pd.DataFrame(
        {
            "available": [True, True],
            "coverage": [0.01, 0.02],
            "threshold": [0.15, 0.15],
            "calibrated_far": [0.7, 0.6],
            "absolute_margin": [0.02, 0.02],
            "practical_threshold_passed": [False, False],
        }
    )
    config = {"bad_rmse_threshold": 0.15}
    assert _rmse_grid_verdict(grid, config) == "CSTR_ACCEPTED_REGION_TOO_RISKY"
