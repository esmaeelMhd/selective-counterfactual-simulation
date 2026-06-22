from __future__ import annotations

from pathlib import Path

import pandas as pd


def test_comparator_evaluation_modes_and_flags() -> None:
    path = Path("results/v2_comparator_fairness/evaluation/comparator_fairness_by_row.csv")
    assert path.exists()
    by_row = pd.read_csv(path)
    required = {
        "system_id",
        "seed",
        "model_id",
        "badness_target",
        "coverage",
        "comparator_mode",
        "calibrated_judge_id",
        "baseline_judge_id",
        "calibrated_far",
        "baseline_far",
        "absolute_margin",
        "relative_margin",
        "calibrated_accepted_count",
        "baseline_accepted_count",
        "calibrated_false_accept_count",
        "baseline_false_accept_count",
        "uses_test_labels_for_baseline_selection",
        "is_deployable_comparator",
        "is_diagnostic_comparator",
    }
    assert required.issubset(by_row.columns)
    expected_modes = {
        "row_wise_strongest_baseline_envelope",
        "global_calibration_selected_baseline",
        "per_system_calibration_selected_baseline",
        "per_system_target_calibration_selected_baseline",
        "best_calibrated_family_vs_per_system_target_baseline",
    }
    assert expected_modes.issubset(set(by_row["comparator_mode"]))
    row_wise = by_row[by_row["comparator_mode"] == "row_wise_strongest_baseline_envelope"]
    assert row_wise["uses_test_labels_for_baseline_selection"].all()
    assert row_wise["is_diagnostic_comparator"].all()
    assert not row_wise["is_deployable_comparator"].any()
    deployable = by_row[by_row["is_deployable_comparator"]]
    assert not deployable["uses_test_labels_for_baseline_selection"].any()
    assert {"bad_rmse", "bad_event"}.issubset(set(by_row["badness_target"]))
