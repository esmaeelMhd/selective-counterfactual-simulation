from __future__ import annotations

import json
from pathlib import Path

import yaml


def test_public_benchmark_does_not_expand_claim_or_evidence() -> None:
    config = yaml.safe_load(Path("configs/status/public_benchmark_v1_2.yaml").read_text(encoding="utf-8"))
    manifest = json.loads(Path("results/current_status/evidence_manifest/current_evidence_manifest.json").read_text(encoding="utf-8"))
    package = json.loads(Path("results/public_benchmark_v1_2/public_package_manifest.json").read_text(encoding="utf-8"))

    assert config["expansion_policy"]["scientific_expansion_allowed"] is False
    assert package["scientific_claim_changed"] is False
    assert package["allowed_claim"] == config["allowed_claim"]["text"]
    assert manifest["controlling_claim_text"] == config["allowed_claim"]["text"]
    controlling_text = yaml.safe_dump(config["source_artifacts"]).lower()
    assert "rssm" not in controlling_text
    assert "heat_exchanger" not in controlling_text
    assert "third-system" not in controlling_text
    for forbidden_dir in ["api", "frontend", "dashboard", "database"]:
        assert not Path(forbidden_dir).exists()
