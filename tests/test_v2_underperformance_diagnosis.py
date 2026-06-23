from __future__ import annotations

from pathlib import Path

import pandas as pd

from scs.experiments.v2 import diagnose_v2_calibrated_underperformance


def test_v2_underperformance_diagnosis_uses_frozen_artifacts(tmp_path: Path) -> None:
    result = diagnose_v2_calibrated_underperformance(
        "results/v2_scientific_strengthening/frozen_protocol",
        tmp_path / "diagnosis",
    )
    assert result["verdict"] == "UNDERPERFORMANCE_DIAGNOSED"
    assert result["decision_gate"] == "NO_METHOD_CLAIM_BENCHMARK_ONLY"
    assert result["moving_baseline_comparator"] is True
    assert result["dominant_baseline"] == "conformal_risk_threshold"
    for name in [
        "primary_vs_baseline.csv",
        "baseline_winner_counts.csv",
        "judge_far_ranking.csv",
        "judge_auc_by_target.csv",
        "label_balance.csv",
        "oracle_gap_summary.csv",
        "underperformance_diagnosis_summary.json",
    ]:
        assert (tmp_path / "diagnosis" / name).exists(), name
        assert (tmp_path / "diagnosis" / name).stat().st_size > 0, name
    primary = pd.read_csv(tmp_path / "diagnosis" / "primary_vs_baseline.csv")
    assert {"mean_primary_far", "mean_baseline_far", "mean_absolute_margin"}.issubset(primary.columns)


def test_v2_underperformance_report_is_regenerated_from_csv(tmp_path: Path) -> None:
    diagnose_v2_calibrated_underperformance(
        "results/v2_scientific_strengthening/frozen_protocol",
        tmp_path / "diagnosis",
    )
    canonical = Path("reports/v2_calibrated_underperformance_diagnosis.md")
    text = canonical.read_text(encoding="utf-8")
    assert "UNDERPERFORMANCE_DIAGNOSED" in text
    assert "conformal_risk_threshold" in text
    assert "NO_METHOD_CLAIM_BENCHMARK_ONLY" in text
