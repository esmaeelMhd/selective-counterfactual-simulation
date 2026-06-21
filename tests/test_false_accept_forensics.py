from __future__ import annotations

import pandas as pd

from scs.experiments.effect_audit import _accepted_rows_for_family_best, _false_accept_group_summary, _tag_false_accepts


def _forensics_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "system_id": ["cstr"] * 4,
            "scenario_id": [f"s_{i}" for i in range(4)],
            "model_id": ["m"] * 4,
            "scenario_type": ["feed"] * 4,
            "bad_rmse_label": [False, True, True, False],
            "rmse": [0.1, 0.16, 0.9, 0.05],
            "mae": [0.1, 0.1, 0.5, 0.05],
            "max_abs_error": [0.2, 0.2, 1.0, 0.1],
            "final_state_error": [0.1, 0.1, 0.5, 0.05],
            "support_distance": [0.1, 0.1, 2.0, 0.1],
            "uncertainty_score": [0.1, 0.1, 2.0, 0.1],
            "disagreement_score": [0.1, 0.1, 2.0, 0.1],
            "invariant_residual": [0.1, 0.1, 2.0, 0.1],
            "repair_amount": [0.0, 0.0, 1.0, 0.0],
            "risk_rank_normalized_linear": [0.1, 0.2, 0.9, 0.3],
            "risk_calibration_selected_candidate_ranker": [0.1, 0.2, 0.9, 0.4],
        }
    )


def test_false_accepts_are_identified_tagged_and_grouped() -> None:
    table = _forensics_table()
    accepted = _accepted_rows_for_family_best(table, ["rank_normalized_linear", "calibration_selected_candidate_ranker"], 0.5)
    false_accepts = accepted[accepted["bad_rmse_label"]].copy()
    tagged = _tag_false_accepts(false_accepts, table, threshold=0.15)
    assert len(tagged) == 1
    assert "NEAR_THRESHOLD_FAILURE" in tagged.iloc[0]["tags"]
    grouped = _false_accept_group_summary(tagged, ["system_id", "model_id", "scenario_type"])
    assert grouped.iloc[0]["accepted_bad_count"] == 1
    assert grouped.iloc[0]["worst_bad_rmse"] == 0.16
