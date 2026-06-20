from __future__ import annotations

from pathlib import Path

import pandas as pd

from scs.reports.failure_analysis import analyze_benchmark_sanity


def test_benchmark_sanity_validates_distribution_and_label_checks(tmp_path: Path, failure_results: Path, failure_table_path: Path) -> None:
    summary = analyze_benchmark_sanity(failure_results, failure_table_path, tmp_path)
    checks = pd.read_csv(tmp_path / "benchmark_sanity.csv")
    assert summary["verdict"] in {"VALID_BENCHMARK", "WEAK_BENCHMARK", "INVALID_BENCHMARK"}
    assert summary["ood_distribution_differs"] is True
    assert summary["bad_labels_nondegenerate"] is True
    assert checks["passed"].any()


def test_benchmark_sanity_detects_degenerate_bad_labels(tmp_path: Path, failure_results: Path, failure_table_path: Path) -> None:
    table = pd.read_csv(failure_table_path)
    table["rmse"] = 1.0
    table["bad_rmse_label"] = True
    degenerate = tmp_path / "degenerate.csv"
    table.to_csv(degenerate, index=False)
    summary = analyze_benchmark_sanity(failure_results, degenerate, tmp_path / "out")
    assert summary["bad_labels_nondegenerate"] is False
