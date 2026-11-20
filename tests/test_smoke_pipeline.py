from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def _tracked_smoke_bytes(root: Path) -> dict[Path, bytes]:
    saved: dict[Path, bytes] = {}
    smoke_dir = root / "results" / "smoke_two_tank"
    report = root / "reports" / "smoke_report.md"
    if smoke_dir.exists():
        for path in smoke_dir.rglob("*"):
            if path.is_file():
                saved[path] = path.read_bytes()
    if report.is_file():
        saved[report] = report.read_bytes()
    return saved


def test_smoke_pipeline_creates_required_artifacts() -> None:
    root = Path(__file__).resolve().parents[1]
    saved = _tracked_smoke_bytes(root)
    try:
        result = subprocess.run(
            ["python", "scripts/run_smoke.py"],
            cwd=root,
            text=True,
            capture_output=True,
            timeout=240,
            check=False,
        )
        assert result.returncode == 0, result.stderr + result.stdout

        output_dir = root / "results" / "smoke_two_tank"
        required = [
            output_dir / "data_summary.json",
            output_dir / "model_metrics.csv",
            output_dir / "scenario_scores.csv",
            output_dir / "risk_coverage.csv",
            output_dir / "risk_coverage.png",
            output_dir / "summary.json",
            root / "reports" / "smoke_report.md",
        ]
        for path in required:
            assert path.exists(), path
            assert path.stat().st_size > 0, path

        risk = pd.read_csv(output_dir / "risk_coverage.csv")
        required_columns = {
            "system_id",
            "model_id",
            "judge_id",
            "coverage",
            "false_accept_rate",
            "accepted_count",
            "false_accept_count",
            "mean_error_accepted",
            "mean_error_rejected",
            "threshold",
        }
        assert required_columns <= set(risk.columns)
        assert risk["coverage"].nunique() >= 5
        assert risk["judge_id"].nunique() >= 5
        assert np.isfinite(risk.select_dtypes(include=[float, int]).to_numpy()).all()
        for judge_id in ["combined_linear", "support_only", "uncertainty_only", "random_baseline"]:
            assert judge_id in set(risk["judge_id"])
    finally:
        for path, payload in saved.items():
            path.write_bytes(payload)

