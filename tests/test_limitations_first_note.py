from __future__ import annotations

from pathlib import Path

from scs.experiments.technical_note_package import EXACT_CLAIM_SENTENCE, scan_forbidden_claim_language


def test_limitations_first_note_contains_required_claims_and_caveats() -> None:
    text = Path("docs/technical_note_limitations_first.md").read_text(encoding="utf-8")

    assert EXACT_CLAIM_SENTENCE in text
    for sentence in [
        "This is not safety certification.",
        "This is not a product-ready digital twin.",
        "This is not a claim of general simulator reliability.",
        "This is not high-coverage reliability.",
        "This is not evidence for RSSM or a third system.",
    ]:
        assert sentence in text
    assert "CSTR is positive at low coverage, but the margins are small" in text
    assert "MARK_REPAIR_DIAGNOSTIC_ONLY_FOR_CSTR" in text
    assert "repair_amount is diagnostic-only for CSTR" in text


def test_limitations_first_note_claim_language_is_guarded() -> None:
    scan = scan_forbidden_claim_language(["docs/technical_note_limitations_first.md"], ["strong support", "safety certification", "high-coverage reliability"])

    assert scan["violations"] == []
    assert any(item["phrase"] == "safety certification" for item in scan["allowed_mentions"])
