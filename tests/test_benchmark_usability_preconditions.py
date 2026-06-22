from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scs.experiments.benchmark_usability import load_usability_config, verify_benchmark_usability_preconditions


def test_usability_config_preserves_claim_and_policy() -> None:
    config = load_usability_config("configs/status/benchmark_usability_v1_1.yaml")

    assert config["allowed_claim"]["label"] == "WEAK_LOW_COVERAGE_BENCHMARK"
    assert config["expansion_policy"]["scientific_expansion_allowed"] is False
    assert config["expansion_policy"]["usability_expansion_allowed"] is True
    assert "RSSM" in config["forbidden_research_expansion"]


def test_usability_preconditions_write_hashes(tmp_path: Path) -> None:
    result = verify_benchmark_usability_preconditions(
        "configs/status/benchmark_usability_v1_1.yaml",
        tmp_path / "preconditions",
        report_output=tmp_path / "report.md",
    )
    hashes = json.loads((tmp_path / "preconditions" / "source_artifact_hashes.json").read_text(encoding="utf-8"))

    assert result["verdict"] == "READY_FOR_BENCHMARK_USABILITY_EXPANSION"
    assert hashes["artifacts"]


def test_scientific_expansion_true_fails(tmp_path: Path) -> None:
    config = yaml.safe_load(Path("configs/status/benchmark_usability_v1_1.yaml").read_text(encoding="utf-8"))
    config["expansion_policy"]["scientific_expansion_allowed"] = True
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="scientific expansion"):
        load_usability_config(path)
