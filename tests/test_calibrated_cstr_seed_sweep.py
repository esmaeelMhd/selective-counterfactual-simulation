from __future__ import annotations

from pathlib import Path

import pandas as pd

from conftest import write_tiny_calibrated_cstr_config
from scs.experiments.calibrated import run_calibrated_seed_sweep


def test_tiny_cstr_seed_sweep_keeps_all_seeds(tmp_path: Path) -> None:
    config = write_tiny_calibrated_cstr_config(tmp_path / "tiny_cstr.yaml")
    summary = run_calibrated_seed_sweep(config, [0, 1], tmp_path / "seed_sweep")
    per_seed = pd.read_csv(tmp_path / "seed_sweep" / "seed_sweep_calibrated_summary.csv")
    all_risk = pd.read_csv(tmp_path / "seed_sweep" / "calibrated_risk_coverage_all.csv")
    assert set(per_seed["seed"]) == {0, 1}
    assert set(all_risk["seed"]) == {0, 1}
    assert summary["verdict"] in {"ROBUST_LOW_COVERAGE", "UNSTABLE", "NO_ROBUST_IMPROVEMENT", "INVALID_DUE_TO_LEAKAGE"}
    assert (tmp_path / "seed_sweep" / "seed_0" / "calibrated_judge_summary.json").exists()
    assert (tmp_path / "seed_sweep" / "seed_1" / "judge_provenance.json").exists()
