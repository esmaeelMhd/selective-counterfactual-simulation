from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from conftest import write_tiny_calibrated_cstr_config
from scs.experiments.calibrated import run_calibrated_judge


def test_tiny_calibrated_cstr_pipeline_outputs_required_artifacts(tmp_path: Path) -> None:
    config = write_tiny_calibrated_cstr_config(tmp_path / "tiny_cstr.yaml")
    output = tmp_path / "calibrated_cstr"
    summary = run_calibrated_judge(config, output, report_path=tmp_path / "calibrated_cstr_report.md")
    required = [
        "calibration_table.csv",
        "test_table.csv",
        "judge_provenance.json",
        "calibration_selection.csv",
        "calibrated_risk_coverage.csv",
        "test_comparison.csv",
        "low_coverage_summary.csv",
        "calibrated_judge_summary.json",
        "calibrated_risk_coverage.png",
    ]
    for name in required:
        assert (output / name).exists()
    calibration = pd.read_csv(output / "calibration_table.csv")
    test = pd.read_csv(output / "test_table.csv")
    assert set(calibration["role"]) == {"judge_calibration"}
    assert set(test["role"]) == {"judge_test"}
    assert set(calibration["scenario_id"]).isdisjoint(set(test["scenario_id"]))
    provenance = json.loads((output / "judge_provenance.json").read_text(encoding="utf-8"))
    assert all(item["used_test_labels_during_fit"] is False for item in provenance["judges"])
    risk = pd.read_csv(output / "calibrated_risk_coverage.csv")
    oracle = risk[risk["judge_id"] == "oracle_error_rank"]
    assert not oracle.empty
    assert oracle["is_oracle"].all()
    assert summary["verdict"] in {"SUPPORTED_LOW_COVERAGE", "MIXED", "NO_IMPROVEMENT_OVER_SINGLE_SIGNAL", "INVALID_DUE_TO_LEAKAGE"}
