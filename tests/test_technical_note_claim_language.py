from __future__ import annotations

from pathlib import Path

from scs.experiments.technical_note_package import scan_forbidden_claim_language


def test_positive_forbidden_phrase_is_flagged(tmp_path: Path) -> None:
    path = tmp_path / "bad.md"
    path.write_text("# Result\n\nThis has strong support for high-coverage reliability.\n", encoding="utf-8")
    scan = scan_forbidden_claim_language([path], ["strong support", "high-coverage reliability"])

    assert {item["phrase"] for item in scan["violations"]} >= {"strong support", "high-coverage reliability"}


def test_forbidden_phrase_under_non_claims_is_allowed(tmp_path: Path) -> None:
    path = tmp_path / "non_claims.md"
    path.write_text("# Non-claims\n\nThis is not safety certification.\n", encoding="utf-8")
    scan = scan_forbidden_claim_language([path], ["safety certification"])

    assert scan["violations"] == []
    assert scan["allowed_mentions"]


def test_forbidden_phrase_under_limitations_is_allowed(tmp_path: Path) -> None:
    path = tmp_path / "limitations.md"
    path.write_text("# Limitations\n\nThis is not high-coverage reliability.\n", encoding="utf-8")
    scan = scan_forbidden_claim_language([path], ["high-coverage reliability"])

    assert scan["violations"] == []
    assert scan["allowed_mentions"]
