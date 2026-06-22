from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scs.experiments.public_benchmark import load_public_config, verify_public_benchmark_preconditions


def test_public_benchmark_config_preserves_claim_and_policy() -> None:
    config = load_public_config("configs/status/public_benchmark_v1_2.yaml")

    assert config["allowed_claim"]["text"] == "A weak but positive low-coverage result under the frozen protocol."
    assert config["expansion_policy"]["scientific_expansion_allowed"] is False
    assert config["expansion_policy"]["public_packaging_allowed"] is True
    assert "RSSM" in config["forbidden_research_expansion"]
    assert config["required_public_hook"]["text"].startswith("A benchmark for testing")


def test_public_preconditions_write_hash_manifest(tmp_path: Path) -> None:
    result = verify_public_benchmark_preconditions(
        "configs/status/public_benchmark_v1_2.yaml",
        tmp_path / "preconditions",
        report_output=tmp_path / "report.md",
    )
    hashes = json.loads((tmp_path / "preconditions" / "source_artifact_hashes.json").read_text(encoding="utf-8"))

    assert result["verdict"] == "READY_FOR_PUBLIC_BENCHMARK_PACKAGING"
    assert "README.md" in hashes["artifacts"]
    assert "docs/benchmark_card.md" in hashes["artifacts"]


def test_public_preconditions_reject_scientific_expansion(tmp_path: Path) -> None:
    config = yaml.safe_load(Path("configs/status/public_benchmark_v1_2.yaml").read_text(encoding="utf-8"))
    config["expansion_policy"]["scientific_expansion_allowed"] = True
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="scientific expansion"):
        load_public_config(path)
