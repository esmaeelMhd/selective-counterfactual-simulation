from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from scs.experiments.v2_public_hardening import run_public_benchmark


def _sha(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _assert_public_outputs(out: Path) -> None:
    for name in [
        "risk_coverage.csv",
        "model_metrics.csv",
        "event_metrics.csv",
        "accepted_false_accepts.csv",
        "benchmark_summary.json",
        "risk_coverage.png",
        "benchmark_report.md",
    ]:
        assert (out / name).exists()
        assert (out / name).stat().st_size > 0
    risk = pd.read_csv(out / "risk_coverage.csv")
    assert {"bad_rmse", "bad_event", "bad_rmse_or_event"}.issubset(set(risk["badness_target"]))
    accepted = pd.read_csv(out / "accepted_false_accepts.csv")
    assert "false_accept" in accepted.columns
    report = (out / "benchmark_report.md").read_text(encoding="utf-8")
    assert "This benchmark run does not update the repository's current scientific claim." in report


def test_run_benchmark_custom_model(tmp_path: Path) -> None:
    manifest = "results/current_status/evidence_manifest/current_evidence_manifest.json"
    before = _sha(manifest)
    out = tmp_path / "custom"
    result = run_public_benchmark(
        out,
        custom_model="examples/custom_model_example.py:DampedLinearUserModel",
        command="custom",
    )
    assert result["verdict"] == "PUBLIC_BENCHMARK_RUN_COMPLETE"
    assert result["current_claim_updated"] is False
    _assert_public_outputs(out)
    risk = pd.read_csv(out / "risk_coverage.csv")
    assert risk["is_custom_model"].all()
    assert _sha(manifest) == before


def test_run_benchmark_builtin_models(tmp_path: Path) -> None:
    out = tmp_path / "builtin"
    result = run_public_benchmark(out, models=["linear_narx", "mlp_state_space"], command="builtin")
    assert result["verdict"] == "PUBLIC_BENCHMARK_RUN_COMPLETE"
    _assert_public_outputs(out)
    risk = pd.read_csv(out / "risk_coverage.csv")
    assert risk["is_builtin_model"].all()
    assert set(risk["model_id"]) == {"linear_narx", "mlp_state_space"}
