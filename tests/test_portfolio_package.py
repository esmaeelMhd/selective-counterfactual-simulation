from __future__ import annotations

from pathlib import Path


def test_one_page_summary_under_900_words() -> None:
    words = Path("docs/one_page_project_summary.md").read_text(encoding="utf-8").split()
    assert len(words) < 900


def test_portfolio_summary_does_not_exaggerate() -> None:
    text = Path("docs/portfolio_summary.md").read_text(encoding="utf-8").lower()
    assert "breakthrough" not in text
    assert "safety certification" not in text
    assert "product-ready" not in text


def test_claim_audit_table_has_supported_weak_false_and_forbidden_claims() -> None:
    text = Path("docs/claim_audit_table.md").read_text(encoding="utf-8")
    for token in ["supported", "weak positive", "false", "forbidden"]:
        assert token in text
    for claim in [
        "combined_linear works",
        "calibrated low-coverage works on TwoTank",
        "calibrated low-coverage weakly replicates on CSTR",
        "repair_amount is universal",
        "invariant_residual is informative on CSTR",
        "general simulator reliability",
        "product readiness",
    ]:
        assert claim in text


def test_reproducibility_card_includes_commands() -> None:
    text = Path("docs/reproducibility_card.md").read_text(encoding="utf-8")
    for command in ['pip install -e ".[dev]"', "pytest -q", "python scripts/run_smoke.py", "check_technical_note_package.py"]:
        assert command in text
