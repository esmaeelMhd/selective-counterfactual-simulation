from __future__ import annotations

from pathlib import Path

from scs.experiments.v2_public_hardening import verify_public_benchmark_hardening_preconditions


def test_v2_public_benchmark_preconditions_ready(tmp_path: Path) -> None:
    result = verify_public_benchmark_hardening_preconditions(
        "configs/v2/v2_public_benchmark_hardening.yaml",
        tmp_path / "preconditions",
    )
    assert result["verdict"] == "READY_FOR_PUBLIC_BENCHMARK_HARDENING"
    assert result["missing_source_artifacts"] == []
    assert result["source_artifact_modified"] is False
    assert (tmp_path / "preconditions" / "source_artifact_hashes.json").exists()
