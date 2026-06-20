from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def test_full_v0_pipeline_runs_three_systems() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["python", "scripts/evaluate_selective.py", "--config", "configs/experiments/full_v0.yaml"],
        cwd=root,
        text=True,
        capture_output=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout

    output_dir = root / "results" / "full_v0"
    required = [
        output_dir / "risk_coverage.csv",
        output_dir / "risk_coverage.png",
        output_dir / "risk_coverage_two_tank.png",
        output_dir / "risk_coverage_cstr.png",
        output_dir / "risk_coverage_heat_exchanger.png",
        output_dir / "summary.json",
        root / "reports" / "full_v0_report.md",
    ]
    for path in required:
        assert path.exists(), path
        assert path.stat().st_size > 0, path

    risk = pd.read_csv(output_dir / "risk_coverage.csv")
    assert set(risk["system_id"]) == {"two_tank", "cstr", "heat_exchanger"}
    assert risk["model_id"].nunique() >= 3
    assert risk["judge_id"].nunique() >= 6
    assert np.isfinite(risk.select_dtypes(include=[float, int]).to_numpy()).all()

    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["claim_status"]["result"] in {"SUPPORTED", "MIXED", "NOT SUPPORTED"}
    assert set(summary["systems"]) == {"two_tank", "cstr", "heat_exchanger"}
