from __future__ import annotations

import json
from pathlib import Path


def test_v0_smoke_artifact_contract_if_present() -> None:
    root = Path("results/smoke_two_tank")
    if not root.exists():
        return
    required = [
        root / "data_summary.json",
        root / "model_metrics.csv",
        root / "scenario_scores.csv",
        root / "risk_coverage.csv",
        root / "risk_coverage.png",
        root / "summary.json",
        Path("reports/smoke_report.md"),
    ]
    for path in required:
        assert path.exists(), path
        assert path.stat().st_size > 0, path
    json.loads((root / "summary.json").read_text())

