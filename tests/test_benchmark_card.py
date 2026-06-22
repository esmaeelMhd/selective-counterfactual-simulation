from __future__ import annotations

from pathlib import Path

from scs.experiments.benchmark_usability import build_benchmark_card
from scs.experiments.technical_note_package import scan_forbidden_claim_language


def test_benchmark_card_contains_required_sections_and_boundaries(tmp_path: Path) -> None:
    output = tmp_path / "benchmark_card.md"
    build_benchmark_card(
        "configs/status/benchmark_usability_v1_1.yaml",
        "results/current_status/evidence_manifest/current_evidence_manifest.json",
        output,
    )
    text = output.read_text(encoding="utf-8")

    for section in [
        "## Intended use",
        "## Non-intended use",
        "## What counts as a false accept",
        "## Known weaknesses",
        "## How to add a custom model",
        "## Claim boundaries",
    ]:
        assert section in text
    assert "Current evidence is weak-positive and low-coverage only." in text
    assert "repair_amount is diagnostic-only for CSTR" in text
    scan = scan_forbidden_claim_language([output], ["safety certification", "high-coverage reliability", "strong support"])
    assert scan["violations"] == []
