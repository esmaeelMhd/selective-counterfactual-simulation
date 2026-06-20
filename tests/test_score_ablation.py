from __future__ import annotations

from pathlib import Path

import pandas as pd

from scs.reports.failure_analysis import analyze_score_ablation


def test_score_ablation_includes_signal_removals_and_grouped_validation(tmp_path: Path, failure_table_path: Path) -> None:
    summary = analyze_score_ablation(failure_table_path, tmp_path)
    result = pd.read_csv(tmp_path / "score_ablation.csv")
    expected = {
        "combined_linear",
        "combined_without_support",
        "combined_without_uncertainty",
        "combined_without_disagreement",
        "combined_without_invariant",
        "combined_without_repair",
        "rank_normalized_combined",
        "logistic_error_classifier",
        "isotonic_calibrated_score",
    }
    assert expected <= set(result["score"])
    assert summary["validation_scheme"] == "grouped cross-validation by scenario_id"
    assert result["win_rate_vs_best_simple"].dropna().between(0.0, 1.0).all()
    report = Path("reports/score_ablation.md").read_text(encoding="utf-8")
    assert "grouped validation" in report or "grouped cross-validation" in report
