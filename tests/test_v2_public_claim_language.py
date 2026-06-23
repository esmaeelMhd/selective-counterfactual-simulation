from __future__ import annotations

from scs.experiments.v2_public_hardening import scan_public_claim_language


FORBIDDEN = ["robust calibrated-refusal method", "safety certification", "product readiness"]


def test_positive_method_success_claim_is_flagged() -> None:
    hits = scan_public_claim_language("This is a robust calibrated-refusal method.", FORBIDDEN)
    assert "robust calibrated-refusal method" in hits


def test_same_phrase_under_forbidden_section_allowed() -> None:
    text = "## Forbidden claims\n- robust calibrated-refusal method\n- safety certification\n"
    assert scan_public_claim_language(text, FORBIDDEN) == []


def test_safety_and_product_claims_flagged() -> None:
    hits = scan_public_claim_language("This provides safety certification and product readiness.", FORBIDDEN)
    assert "safety certification" in hits
    assert "product readiness" in hits


def test_positive_claim_in_body_bullet_is_flagged() -> None:
    hits = scan_public_claim_language("- robust calibrated-refusal method", FORBIDDEN)
    assert "robust calibrated-refusal method" in hits


def test_negated_claim_boundary_sentence_allowed() -> None:
    text = "This benchmark does not provide safety certification or product readiness."
    assert scan_public_claim_language(text, FORBIDDEN) == []


def test_bold_not_supported_context_allows_following_bullets() -> None:
    text = "**What is not supported:**\n- safety certification\n- product readiness\n"
    assert scan_public_claim_language(text, FORBIDDEN) == []
