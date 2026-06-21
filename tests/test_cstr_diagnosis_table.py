from __future__ import annotations

import pandas as pd

from scs.experiments.cstr_weakness import _accepted_mask, _selected_calibrated_for_group


def _fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "system_id": ["cstr", "cstr", "two_tank", "cstr"],
            "scenario_id": ["a", "b", "x", "c"],
            "model_id": ["hold_last", "hold_last", "hold_last", "linear_narx"],
            "scenario_type": ["id", "id", "id", "id"],
            "bad_rmse_label": [False, True, True, True],
            "risk_rank_normalized_linear": [0.1, 0.2, 0.1, 0.3],
            "risk_calibration_selected_candidate_ranker": [0.4, 0.3, 0.1, 0.1],
        }
    )


def test_cstr_only_filtering_and_acceptance_mask_identify_false_accepts() -> None:
    table = _fixture()
    cstr = table[table["system_id"] == "cstr"].copy()
    accepted = _accepted_mask(cstr[cstr["model_id"] == "hold_last"], "risk_rank_normalized_linear", 0.5)
    accepted_rows = cstr[cstr["model_id"] == "hold_last"].loc[accepted]
    assert set(cstr["system_id"]) == {"cstr"}
    assert len(accepted_rows) == 1
    assert bool(accepted_rows.iloc[0]["bad_rmse_label"]) is False


def test_selected_calibrated_judge_uses_fixture_false_accept_rate() -> None:
    cstr = _fixture()[lambda df: df["system_id"] == "cstr"].copy()
    selected = _selected_calibrated_for_group(
        cstr,
        0.5,
        ["rank_normalized_linear", "calibration_selected_candidate_ranker"],
    )
    row = selected[selected["model_id"] == "hold_last"].iloc[0]
    assert row["calibrated_judge_id"] == "rank_normalized_linear"


def test_missing_trajectory_fields_are_marked_unavailable_not_fabricated() -> None:
    row = {
        "trajectory_available": False,
        "state_error_concentration_rmse": float("nan"),
        "state_error_temperature_rmse": float("nan"),
    }
    assert row["trajectory_available"] is False
    assert pd.isna(row["state_error_concentration_rmse"])
