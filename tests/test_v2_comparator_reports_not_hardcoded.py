from __future__ import annotations

from pathlib import Path

import pandas as pd

from scs.experiments.v2_comparator import report_contains_literal_numbers


def test_comparator_selection_report_contains_actual_csv_value() -> None:
    report = Path("reports/v2_comparator_selection.md")
    csv = Path("results/v2_comparator_fairness/comparator_selection/global_baseline_selection.csv")
    assert report_contains_literal_numbers(report, csv, "selected_judge_id")


def test_report_value_check_changes_when_csv_changes(tmp_path: Path) -> None:
    report = tmp_path / "report.md"
    csv = tmp_path / "values.csv"
    pd.DataFrame({"metric": [0.123456]}).to_csv(csv, index=False)
    report.write_text("metric is 0.123456", encoding="utf-8")
    assert report_contains_literal_numbers(report, csv, "metric")
    pd.DataFrame({"metric": [0.654321]}).to_csv(csv, index=False)
    assert not report_contains_literal_numbers(report, csv, "metric")
