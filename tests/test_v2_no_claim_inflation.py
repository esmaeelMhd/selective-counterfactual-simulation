from __future__ import annotations

from pathlib import Path


def test_v2_docs_do_not_inflate_claim_after_negative_gate() -> None:
    decision = Path("reports/v2_scientific_decision_gate.md").read_text(encoding="utf-8")
    assert "NO_METHOD_CLAIM_BENCHMARK_ONLY" in decision
    summary = Path("docs/v2/v2_scientific_summary.md").read_text(encoding="utf-8")
    assert "This repository is a benchmark only" in summary
    assert "safety certification" not in summary.lower()
    assert "trusted simulator" not in summary.lower()
    assert "validated digital twin" not in summary.lower()


def test_v2_claim_audit_table_contains_required_claims() -> None:
    text = Path("docs/v2/v2_claim_audit_table.md").read_text(encoding="utf-8")
    assert "| Claim | Status | Evidence | Allowed wording |" in text
    for claim in [
        "calibrated refusal works generally",
        "calibrated refusal works at low coverage",
        "event-risk refusal works",
        "signals are universal",
        "repair_amount is universal",
        "benchmark exposes system dependence",
    ]:
        assert claim in text
