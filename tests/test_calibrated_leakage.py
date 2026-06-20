from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scs.validators.calibrated import BestSingleSignalJudge, RankNormalizedLinearJudge, make_calibrated_judges


def _table(role: str, split: str, support_bad: bool) -> pd.DataFrame:
    rmse = np.array([0.02, 0.04, 0.25, 0.30])
    support = np.array([0.0, 0.1, 0.9, 1.0]) if support_bad else np.array([0.9, 1.0, 0.0, 0.1])
    disagreement = np.array([0.9, 1.0, 0.0, 0.1]) if support_bad else np.array([0.0, 0.1, 0.9, 1.0])
    return pd.DataFrame(
        {
            "scenario_id": [f"{split}_{i}" for i in range(4)],
            "role": [role] * 4,
            "split": [split] * 4,
            "support_distance": support,
            "uncertainty_score": np.zeros(4),
            "disagreement_score": disagreement,
            "invariant_residual": rmse,
            "repair_amount": np.zeros(4),
            "rmse": rmse,
            "bad_rmse_label": rmse > 0.15,
        }
    )


def test_fit_rejects_test_rows_for_all_calibrated_judges() -> None:
    table = _table("judge_test", "judge_test_id", True)
    signals = ["support_distance", "uncertainty_score", "disagreement_score", "invariant_residual", "repair_amount"]
    for judge in make_calibrated_judges([0.05], 0.15):
        with pytest.raises(ValueError):
            judge.fit(table, signals, "rmse", "bad_rmse_label")


def test_best_single_signal_is_selected_from_calibration_only() -> None:
    calibration = _table("judge_calibration", "judge_calibration_id", True)
    test = _table("judge_test", "judge_test_id", False)
    signals = ["support_distance", "disagreement_score"]
    judge = BestSingleSignalJudge([0.05, 0.10], 0.15)
    judge.fit(calibration, signals, "rmse", "bad_rmse_label")
    assert judge.provenance()["selected_signal_if_any"] == "support_distance"
    scores = judge.score(test)
    assert scores[0] > scores[-1]


def test_rank_orientation_is_learned_from_calibration_only() -> None:
    calibration = _table("judge_calibration", "judge_calibration_id", False)
    signals = ["support_distance", "disagreement_score"]
    judge = RankNormalizedLinearJudge([0.05], 0.15)
    judge.fit(calibration, signals, "rmse", "bad_rmse_label")
    orientation = judge.provenance()["selected_hyperparameters"]["orientation"]
    assert orientation["support_distance"] == -1
