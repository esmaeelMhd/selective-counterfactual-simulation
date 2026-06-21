from __future__ import annotations

import pandas as pd

from scs.experiments.effect_audit import _effect_verdict, family_best_far_for_table, far_for_table


def _fixture_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model_id": ["m"] * 4,
            "scenario_type": ["s"] * 4,
            "bad_rmse_label": [False, True, True, False],
            "risk_best_single_signal_selected_on_calibration": [0.1, 0.2, 0.3, 0.4],
            "risk_rank_normalized_linear": [0.1, 0.4, 0.5, 0.2],
            "risk_calibration_selected_candidate_ranker": [0.2, 0.3, 0.4, 0.1],
        }
    )


def test_far_margin_and_practical_thresholds_from_fixture() -> None:
    table = _fixture_table()
    baseline = far_for_table(table, "best_single_signal_selected_on_calibration", 0.5)
    calibrated = family_best_far_for_table(
        table,
        ["rank_normalized_linear", "calibration_selected_candidate_ranker"],
        0.5,
    )
    assert baseline["false_accept_rate"] == 0.5
    assert calibrated["false_accept_rate"] == 0.0
    assert baseline["false_accept_rate"] - calibrated["false_accept_rate"] == 0.5


def test_effect_verdict_rules_for_weak_two_system_effect() -> None:
    rows = pd.DataFrame(
        [
            {"system_id": "two_tank", "verdict": "PRACTICALLY_MEANINGFUL", "absolute_margin": 0.1, "seed_win_rate": 1.0},
            {"system_id": "cstr", "verdict": "POSITIVE_BUT_WEAK", "absolute_margin": 0.02, "seed_win_rate": 0.7},
        ]
    )
    config = {"seed_win_threshold_weak": 0.6, "seed_win_threshold_strong": 0.8}
    assert _effect_verdict(rows, config) == "WEAK_TWO_SYSTEM_EFFECT"
