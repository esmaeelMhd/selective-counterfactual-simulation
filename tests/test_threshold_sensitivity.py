from __future__ import annotations

from pathlib import Path

import pandas as pd

from scs.reports.failure_analysis import analyze_threshold_sensitivity


def test_threshold_sensitivity_marks_degenerate_thresholds(tmp_path: Path, failure_table_path: Path) -> None:
    thresholds = [0.05, 0.15, 0.50]
    summary = analyze_threshold_sensitivity(failure_table_path, thresholds, tmp_path)
    result = pd.read_csv(tmp_path / "threshold_sensitivity.csv")
    assert summary["verdict"] in {"ROBUST_TO_THRESHOLD", "THRESHOLD_DEPENDENT", "UNSUPPORTED_ACROSS_THRESHOLDS"}
    assert set(result["threshold"]) == set(thresholds)
    assert "UNAVAILABLE" in set(result[result["threshold"].isin([0.05, 0.50])]["verdict"])
    assert result["false_accept_rate"].dropna().between(0.0, 1.0).all()
