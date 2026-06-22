from __future__ import annotations

import json
from pathlib import Path

import yaml


def test_usability_release_does_not_expand_scientific_claim() -> None:
    config = yaml.safe_load(Path("configs/status/benchmark_usability_v1_1.yaml").read_text(encoding="utf-8"))
    release = json.loads(Path("results/benchmark_usability/release/benchmark_usability_manifest.json").read_text(encoding="utf-8"))

    assert config["expansion_policy"]["scientific_expansion_allowed"] is False
    assert release["scientific_claim_changed"] is False
    assert release["allowed_claim"] == config["allowed_claim"]["text"]


def test_no_forbidden_evidence_or_surface_dirs() -> None:
    config_text = Path("configs/status/benchmark_usability_v1_1.yaml").read_text(encoding="utf-8").lower()
    controlling = yaml.safe_load(Path("configs/status/benchmark_usability_v1_1.yaml").read_text(encoding="utf-8"))["controlling_status"]
    controlling_text = yaml.safe_dump(controlling).lower()

    assert "rssm" not in controlling_text
    assert "heat_exchanger" not in controlling_text
    assert "third system" not in controlling_text
    assert "rssm" in config_text
    for forbidden_dir in ["api", "frontend", "dashboard", "database"]:
        assert not Path(forbidden_dir).exists()
