from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from scs.experiments.public_benchmark import reproduce_main_twotank_result, run_smoke_demo


def test_smoke_demo_runs_and_is_marked_smoke_only(tmp_path: Path) -> None:
    output = tmp_path / "smoke_demo"
    summary = run_smoke_demo(output)

    assert summary["verdict"] == "SMOKE_DEMO_BUILT"
    assert summary["is_smoke_only"] is True
    report = (output / "smoke_demo_report.md").read_text(encoding="utf-8")
    assert "not the full evidence reproduction" in report
    assert (output / "smoke_demo_plot.png").stat().st_size > 0


def test_twotank_reproduction_matches_source_and_preserves_manifest(tmp_path: Path) -> None:
    manifest = Path("results/current_status/evidence_manifest/current_evidence_manifest.json")
    before = hashlib.sha256(manifest.read_bytes()).hexdigest()
    output = tmp_path / "reproduce_twotank"

    summary = reproduce_main_twotank_result(output)
    result = pd.read_csv(output / "twotank_main_result.csv")
    source = pd.read_csv("results/calibrated_two_tank/low_coverage_summary.csv")

    assert summary["verdict"] == "TWOTANK_MAIN_RESULT_REPRODUCED"
    assert set(["system_id", "coverage", "baseline_far", "calibrated_far", "absolute_margin", "source_artifact", "is_reproduction"]).issubset(result.columns)
    for coverage in [0.05, 0.10]:
        actual = result[np.isclose(result["coverage"], coverage)].iloc[0]["absolute_margin"]
        expected = source[np.isclose(source["coverage"], coverage)].iloc[0]["margin"]
        assert abs(actual - expected) <= 1e-6
    assert result[np.isclose(result["coverage"], 0.05)].iloc[0]["absolute_margin"] > 0.17
    assert (output / "twotank_main_result.png").stat().st_size > 0
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == before
