from __future__ import annotations

import pandas as pd

from scs.experiments.v2_comparator import select_judge_from_calibration


def test_test_rows_cannot_be_used_for_deployable_selection() -> None:
    table = pd.DataFrame(
        {
            "role": ["judge_calibration", "judge_test"],
            "bad_label": [False, True],
            "badness_error": [0.0, 1.0],
            "risk_support_only": [0.0, 1.0],
        }
    )
    try:
        select_judge_from_calibration(table, ["support_only"], 0.5)
    except ValueError as exc:
        assert "calibration rows only" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("mixed calibration/test selection must fail")


def test_generated_selection_has_no_test_labels_or_oracle() -> None:
    selection = pd.read_csv("results/v2_comparator_fairness/comparator_selection/per_system_target_baseline_selection.csv")
    assert not selection["uses_test_labels"].any()
    assert set(selection["source_split"]) == {"calibration"}
    assert "oracle_error_rank" not in set(selection["selected_judge_id"])
    evaluation = pd.read_csv("results/v2_comparator_fairness/evaluation/comparator_fairness_by_row.csv")
    deployable = evaluation[evaluation["is_deployable_comparator"]]
    assert not deployable["uses_test_labels_for_baseline_selection"].any()
