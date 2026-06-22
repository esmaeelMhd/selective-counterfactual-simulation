from __future__ import annotations

import pandas as pd

from scs.experiments.v2_comparator import select_judge_from_calibration


def test_calibration_best_can_differ_from_test_best_and_selection_uses_calibration_only() -> None:
    calibration = pd.DataFrame(
        {
            "role": ["judge_calibration"] * 4,
            "bad_label": [False, False, True, True],
            "badness_error": [0.0, 0.0, 1.0, 1.0],
            "risk_alpha": [0.1, 0.2, 0.8, 0.9],
            "risk_beta": [0.7, 0.8, 0.1, 0.2],
        }
    )
    test = pd.DataFrame(
        {
            "role": ["judge_test"] * 4,
            "bad_label": [False, False, True, True],
            "badness_error": [0.0, 0.0, 1.0, 1.0],
            "risk_alpha": [0.9, 0.8, 0.1, 0.2],
            "risk_beta": [0.1, 0.2, 0.8, 0.9],
        }
    )
    selected = select_judge_from_calibration(calibration, ["alpha", "beta"], 0.5)
    test_selected = select_judge_from_calibration(test.assign(role="judge_calibration"), ["alpha", "beta"], 0.5)
    assert selected["selected_judge_id"] == "alpha"
    assert test_selected["selected_judge_id"] == "beta"
    assert selected["selected_judge_id"] != test_selected["selected_judge_id"]


def test_selection_tie_breaker_is_deterministic_and_oracle_forbidden() -> None:
    calibration = pd.DataFrame(
        {
            "role": ["judge_calibration"] * 4,
            "bad_label": [False, True, False, True],
            "badness_error": [0.0, 1.0, 0.0, 1.0],
            "risk_b": [0.1, 0.2, 0.3, 0.4],
            "risk_a": [0.1, 0.2, 0.3, 0.4],
        }
    )
    selected = select_judge_from_calibration(calibration, ["b", "a"], 0.5)
    assert selected["selected_judge_id"] == "a"
    assert selected["tie_breaker_used"] is True
    try:
        select_judge_from_calibration(calibration.assign(risk_oracle_error_rank=0.0), ["oracle_error_rank"], 0.5)
    except ValueError as exc:
        assert "oracle_error_rank" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("oracle selection must fail")
