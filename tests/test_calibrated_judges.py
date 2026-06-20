from __future__ import annotations

import numpy as np
import pandas as pd

from scs.validators.calibrated import REQUIRED_PROVENANCE_KEYS, make_calibrated_judges


def _calibration_table() -> pd.DataFrame:
    n = 12
    rmse = np.linspace(0.02, 0.36, n)
    return pd.DataFrame(
        {
            "scenario_id": [f"cal_{i}" for i in range(n)],
            "role": ["judge_calibration"] * n,
            "split": ["judge_calibration_id"] * n,
            "support_distance": np.linspace(0.0, 1.0, n),
            "uncertainty_score": np.linspace(1.0, 0.0, n),
            "disagreement_score": np.linspace(0.1, 0.8, n),
            "invariant_residual": rmse,
            "repair_amount": np.where(rmse > 0.15, 0.2, 0.0),
            "rmse": rmse,
            "bad_rmse_label": rmse > 0.15,
        }
    )


def test_calibrated_judges_fit_score_and_report_provenance() -> None:
    table = _calibration_table()
    signals = ["support_distance", "uncertainty_score", "disagreement_score", "invariant_residual", "repair_amount"]
    judges = make_calibrated_judges([0.05, 0.10], 0.15)
    for judge in judges:
        judge.fit(table, signals, "rmse", "bad_rmse_label")
        judge.set_test_scenario_hash(["test_0", "test_1"])
        scores = judge.score(table)
        provenance = judge.provenance()
        assert scores.shape == (len(table),)
        assert np.isfinite(scores).all()
        assert set(REQUIRED_PROVENANCE_KEYS) <= set(provenance)
        assert provenance["used_test_labels_during_fit"] is False


def test_unavailable_judge_is_reported_not_silently_skipped() -> None:
    table = _calibration_table()
    table["bad_rmse_label"] = False
    judge = make_calibrated_judges([0.05], 0.15)[2]
    signals = ["support_distance", "uncertainty_score", "disagreement_score", "invariant_residual", "repair_amount"]
    judge.fit(table, signals, "rmse", "bad_rmse_label")
    assert judge.provenance()["available"] is False
    assert judge.provenance()["unavailable_reason"]
