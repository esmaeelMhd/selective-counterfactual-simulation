from __future__ import annotations

from pathlib import Path

import pandas as pd

from conftest import write_tiny_calibrated_config
from scs.experiments.calibrated import run_calibrated_stress


def test_calibrated_stress_outputs_all_thresholds_coverages_and_degeneracy(tmp_path: Path) -> None:
    config = write_tiny_calibrated_config(tmp_path / "tiny.yaml")
    thresholds = [0.15, 999.0]
    coverages = [0.05, 0.10, 0.20]
    summary = run_calibrated_stress(config, thresholds, coverages, [0], tmp_path / "stress")
    result = pd.read_csv(tmp_path / "stress" / "threshold_coverage_results.csv")
    assert set(result["threshold"]) == set(thresholds)
    assert set(result["coverage"]) == set(coverages)
    assert result[result["threshold"] == 999.0]["degenerate"].all()
    assert summary["verdict"] in {
        "ROBUST_LOW_COVERAGE_ONLY",
        "THRESHOLD_DEPENDENT",
        "NO_STABLE_REGION",
        "INVALID_DUE_TO_LEAKAGE",
    }
