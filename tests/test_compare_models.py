from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from conftest import write_tiny_calibrated_config
from scs.experiments.benchmark_usability import compare_models


def test_builtin_model_comparison_runs(tmp_path: Path) -> None:
    config = write_tiny_calibrated_config(tmp_path / "tiny.yaml")
    output = tmp_path / "comparison"

    result = compare_models(config, ["hold_last", "linear_narx"], output)

    assert result["verdict"] == "MODEL_COMPARISON_BUILT"
    for name in [
        "model_comparison.csv",
        "risk_coverage_by_model.csv",
        "risk_coverage_by_model.png",
        "model_comparison_summary.json",
        "model_comparison_report.md",
    ]:
        assert (output / name).exists(), name
        assert (output / name).stat().st_size > 0, name
    table = pd.read_csv(output / "model_comparison.csv")
    assert {"model_id", "rmse_mean", "is_builtin", "is_custom"}.issubset(table.columns)
    assert "local comparison results" in (output / "model_comparison_report.md").read_text(encoding="utf-8")


def test_custom_model_comparison_runs_without_manifest_mutation(tmp_path: Path) -> None:
    manifest = Path("results/current_status/evidence_manifest/current_evidence_manifest.json")
    before = hashlib.sha256(manifest.read_bytes()).hexdigest()
    config = write_tiny_calibrated_config(tmp_path / "tiny.yaml")

    compare_models(
        config,
        ["linear_narx"],
        tmp_path / "comparison_custom",
        custom_model="examples/custom_model_example.py:DampedLinearUserModel",
    )

    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == before
    table = pd.read_csv(tmp_path / "comparison_custom" / "model_comparison.csv")
    assert table["is_custom"].any()
