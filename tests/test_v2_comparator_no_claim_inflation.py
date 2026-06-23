from __future__ import annotations

from pathlib import Path


def test_comparator_docs_do_not_upgrade_claim_or_make_rowwise_deployable() -> None:
    summary = Path("docs/v2/v2_comparator_fairness_summary.md").read_text(encoding="utf-8").lower()
    claim_audit = Path("docs/v2/v2_comparator_claim_audit.md").read_text(encoding="utf-8").lower()
    decision = Path("reports/v2_comparator_fairness_decision_gate.md").read_text(encoding="utf-8")
    assert "row-wise envelope remains diagnostic only" in summary
    assert "not deployable" in claim_audit
    assert "CALIBRATED_TARGET_DEPENDENT" in decision
    forbidden_phrases = [
        "calibrated refusal works generally",
        "safety certification",
        "trusted simulator",
        "product-ready digital twin",
    ]
    for phrase in forbidden_phrases:
        assert phrase not in summary
