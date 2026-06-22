from __future__ import annotations

from pathlib import Path

from scs.experiments.v2_comparator import verify_comparator_preconditions


def test_v2_comparator_preconditions_ready(tmp_path: Path) -> None:
    result = verify_comparator_preconditions(
        "configs/v2/v2_comparator_fairness.yaml",
        tmp_path / "preconditions",
    )
    assert result["verdict"] == "READY_FOR_COMPARATOR_FAIRNESS_AUDIT"
    assert result["v2_decision_ok"] is True
    assert result["diagnostic_only_ok"] is True
    assert result["risk_columns_ok"] is True
    assert result["scenario_csv_non_empty"] is True
    assert (tmp_path / "preconditions" / "precondition_check.json").exists()
    assert (tmp_path / "preconditions" / "source_artifact_hashes.json").exists()
