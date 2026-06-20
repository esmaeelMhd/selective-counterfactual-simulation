from __future__ import annotations

from pathlib import Path

import pandas as pd

from scs.reports.failure_analysis import analyze_coverage_sensitivity


def test_coverage_sensitivity_writes_grid_and_plot(tmp_path: Path, failure_table_path: Path) -> None:
    coverages = [0.10, 0.25, 0.50, 1.00]
    summary = analyze_coverage_sensitivity(failure_table_path, coverages, tmp_path)
    result = pd.read_csv(tmp_path / "coverage_sensitivity.csv")
    assert summary["verdict"] in {"WORKS_AT_LOW_COVERAGE", "WORKS_AT_HIGH_COVERAGE", "FAILS_ACROSS_COVERAGE", "MIXED"}
    assert set(result["coverage"]) == set(coverages)
    assert (tmp_path / "coverage_sensitivity.png").exists()
    assert (tmp_path / "coverage_sensitivity.png").stat().st_size > 0
    report = Path("reports/coverage_sensitivity.md").read_text(encoding="utf-8")
    assert "Oracle is diagnostic only" in report
