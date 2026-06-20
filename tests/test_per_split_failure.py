from __future__ import annotations

from pathlib import Path

import pandas as pd

from scs.reports.failure_analysis import analyze_per_split_failure


def test_per_split_failure_identifies_worst_split_and_excludes_oracle(tmp_path: Path, failure_table_path: Path) -> None:
    summary = analyze_per_split_failure(failure_table_path, tmp_path)
    detail = pd.read_csv(tmp_path / "per_split_failure.csv")
    assert summary["verdict"] in {"BENCHMARK_TOO_EASY", "SPLIT_SPECIFIC_FAILURE", "GLOBAL_FAILURE"}
    assert summary["worst_split_by_error"] == "ood_combined"
    assert "oracle_error_rank" in set(detail["judge_id"])
    report = Path("reports/per_split_failure.md").read_text(encoding="utf-8")
    best_section = report.split("## Best real judge per split", maxsplit=1)[1]
    assert "oracle_error_rank" not in best_section.split("## Where combined_linear failed", maxsplit=1)[0]
