from __future__ import annotations

import pandas as pd

from scs.experiments.cstr_weakness import _dominant_state, _state_summary, _statewise_verdict


def test_statewise_errors_are_summarized_from_state_columns() -> None:
    frame = pd.DataFrame(
        {
            "model_id": ["m", "m"],
            "state_error_concentration_rmse": [1.0, 1.0],
            "state_error_temperature_rmse": [10.0, 10.0],
            "state_error_concentration_max_abs": [1.2, 1.1],
            "state_error_temperature_max_abs": [12.0, 11.0],
            "state_error_concentration_final": [0.5, 0.6],
            "state_error_temperature_final": [5.0, 6.0],
        }
    )
    summary = _state_summary(frame, ["model_id"])
    assert summary.iloc[0]["concentration_rmse_mean"] == 1.0
    assert summary.iloc[0]["temperature_rmse_mean"] == 10.0


def test_dominant_state_rule_uses_normalized_contribution() -> None:
    state, ratio, _, _ = _dominant_state(1.5, 1.0)
    assert state == "concentration"
    assert ratio > 1.0


def test_missing_or_empty_false_accepts_are_unavailable_verdict() -> None:
    frame = pd.DataFrame(columns=["false_accept", "state_error_concentration_rmse", "state_error_temperature_rmse"])
    assert _statewise_verdict(frame) == "TRAJECTORY_DATA_UNAVAILABLE"
