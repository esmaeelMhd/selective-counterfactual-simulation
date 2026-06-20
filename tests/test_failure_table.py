from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scs.reports.failure_analysis import build_failure_table, verify_failed_gate


def test_build_failure_table_schema_and_gate(tmp_path: Path, failure_results: Path) -> None:
    gate = tmp_path / "v0_decision_gate.md"
    gate.write_text("# v0 Decision Gate\n\n## Decision\n\nKILL_OR_DOWNGRADE_CLAIM\n", encoding="utf-8")
    gate_summary = verify_failed_gate(failure_results, gate)
    assert gate_summary["verdict"] == "READY_FOR_FAILURE_ANALYSIS"
    assert gate_summary["expansion_status"] == "BLOCKED"

    output = tmp_path / "failure_analysis" / "failure_table.csv"
    schema = build_failure_table(failure_results, output)
    table = pd.read_csv(output)
    required = {
        "scenario_id",
        "system_id",
        "split",
        "scenario_type",
        "model_id",
        "judge_id",
        "coverage",
        "accepted",
        "risk_score",
        "false_accept",
        "rmse",
        "bad_rmse_label",
        "support_distance",
        "uncertainty_score",
        "disagreement_score",
        "invariant_residual",
        "repair_amount",
        "combined_linear_score",
        "oracle_error_rank_score",
        "random_baseline_score",
        "bad_threshold",
    }
    assert required <= set(table.columns)
    assert schema["row_count"] == len(table)
    assert table["judge_id"].nunique() >= 8
    assert table["coverage"].nunique() >= 3
    assert not table["support_distance"].isna().all()
    assert table["event_error"].isna().all()


def test_build_failure_table_fails_when_artifact_missing(tmp_path: Path, failure_results: Path) -> None:
    (failure_results / "summary.json").unlink()
    with pytest.raises(FileNotFoundError):
        build_failure_table(failure_results, tmp_path / "failure_table.csv")
