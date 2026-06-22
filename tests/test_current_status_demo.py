from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from scs.experiments.benchmark_usability import run_current_status_demo


def test_current_status_demo_runs_and_writes_outputs(tmp_path: Path) -> None:
    manifest = Path("results/current_status/evidence_manifest/current_evidence_manifest.json")
    before = hashlib.sha256(manifest.read_bytes()).hexdigest()
    output = tmp_path / "demo"

    summary = run_current_status_demo("configs/status/benchmark_usability_v1_1.yaml", output)

    assert summary["verdict"] == "DEMO_BUILT"
    for name in ["main_result_table.csv", "risk_coverage.png", "demo_report.md", "demo_summary.json"]:
        path = output / name
        assert path.exists(), path
        assert path.stat().st_size > 0, path
    table = pd.read_csv(output / "main_result_table.csv")
    assert set(["system_id", "coverage", "baseline_far", "calibrated_far", "claim_scope", "is_demo"]).issubset(table.columns)
    report = (output / "demo_report.md").read_text(encoding="utf-8")
    assert "This demo is not the full evidence chain." in report
    assert "The current supported claim remains weak and low-coverage only." in report
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == before
