from __future__ import annotations

from pathlib import Path

import pandas as pd

from scs.reports.failure_analysis import analyze_model_diversity


def test_model_diversity_outputs_oracle_gap_without_deployable_claim(tmp_path: Path, failure_table_path: Path) -> None:
    summary = analyze_model_diversity(failure_table_path, tmp_path)
    diversity = pd.read_csv(tmp_path / "model_diversity.csv")
    oracle = pd.read_csv(tmp_path / "oracle_gap.csv")
    assert summary["verdict"] in {"MODELS_TOO_SIMILAR", "MODELS_TOO_BAD", "REAL_SIGNALS_MISSING", "ORACLE_GAP_SMALL"}
    assert not diversity.empty
    assert not oracle.empty
    report = Path("reports/model_diversity_and_oracle_gap.md").read_text(encoding="utf-8")
    assert "Oracle is diagnostic and not deployable" in report
