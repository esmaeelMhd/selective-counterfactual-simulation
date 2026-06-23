from __future__ import annotations

import pandas as pd

from scs.experiments.v2_comparator import comparator_statistics_verdict


def _effect(rows: list[dict[str, object]]) -> pd.DataFrame:
    base = {
        "comparator_mode": "per_system_target_calibration_selected_baseline",
        "badness_target": "bad_rmse",
        "mean_far_margin": 0.0,
        "positive_ci_excludes_zero": False,
        "seed_win_rate": 0.0,
        "practical_threshold_pass": False,
    }
    return pd.DataFrame([{**base, **row} for row in rows])


def test_statistics_verdict_beats_fair_baseline() -> None:
    frame = _effect(
        [
            {"system_id": "a", "badness_target": "bad_rmse", "mean_far_margin": 0.08, "positive_ci_excludes_zero": True, "seed_win_rate": 0.8, "practical_threshold_pass": True},
            {"system_id": "a", "badness_target": "bad_event", "mean_far_margin": 0.02, "positive_ci_excludes_zero": False, "seed_win_rate": 0.7, "practical_threshold_pass": False},
            {"system_id": "b", "badness_target": "bad_rmse", "mean_far_margin": 0.09, "positive_ci_excludes_zero": True, "seed_win_rate": 0.9, "practical_threshold_pass": True},
            {"system_id": "b", "badness_target": "bad_event", "mean_far_margin": 0.01, "positive_ci_excludes_zero": False, "seed_win_rate": 0.7, "practical_threshold_pass": False},
        ]
    )
    assert comparator_statistics_verdict(frame) == "CALIBRATED_BEATS_FAIR_DEPLOYABLE_BASELINE"


def test_statistics_verdict_target_dependent_and_fails() -> None:
    target_dependent = _effect(
        [
            {"system_id": "a", "badness_target": "bad_rmse", "mean_far_margin": 0.02},
            {"system_id": "a", "badness_target": "bad_event", "mean_far_margin": -0.02},
        ]
    )
    assert comparator_statistics_verdict(target_dependent) == "CALIBRATED_TARGET_DEPENDENT"
    fails = _effect(
        [
            {"system_id": "a", "badness_target": "bad_rmse", "mean_far_margin": -0.01},
            {"system_id": "b", "badness_target": "bad_event", "mean_far_margin": 0.0},
        ]
    )
    assert comparator_statistics_verdict(fails) == "CALIBRATED_FAILS_FAIR_DEPLOYABLE_BASELINE"
