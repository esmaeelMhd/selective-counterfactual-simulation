from __future__ import annotations

import json
from pathlib import Path


def test_benchmark_usability_release_manifest_and_note() -> None:
    manifest = json.loads(Path("results/benchmark_usability/release/benchmark_usability_manifest.json").read_text(encoding="utf-8"))
    note = Path("reports/release_note_v1_1_benchmark_usability.md").read_text(encoding="utf-8")

    assert manifest["release_type"] == "usability_only"
    assert manifest["scientific_claim_changed"] is False
    assert "quickstart demo" in manifest["new_user_features"]
    assert "The scientific claim did not change." in note
    assert "Expansion remains blocked." in note
