from __future__ import annotations

import json
from pathlib import Path

from scs.experiments.current_status import write_current_evidence_manifest_report


def test_current_evidence_manifest_is_weak_no_expansion() -> None:
    manifest = json.loads(Path("results/current_status/evidence_manifest/current_evidence_manifest.json").read_text(encoding="utf-8"))

    assert manifest["status_id"] == "current_evidence_status_v1"
    assert manifest["controlling_claim_label"] == "WEAK_LOW_COVERAGE_CLAIM"
    assert manifest["expansion_allowed"] is False
    assert manifest["allowed_next_action"] == "UPDATE_SIGNAL_SEMANTICS_ONLY"
    assert set(manifest["systems"]) == {"two_tank", "cstr"}
    assert manifest["systems"]["cstr"]["effect_strength"] == "positive_but_weak"
    assert manifest["signal_roles"]["repair_amount"]["cstr_role"] == "diagnostic_only"
    assert manifest["signal_roles"]["invariant_residual"]["cstr_role"] == "informative_refusal_signal"


def test_current_evidence_manifest_report_uses_supplied_values(tmp_path: Path) -> None:
    output = tmp_path / "manifest.md"
    manifest = {
        "practical_utility_decision": "FIXTURE_DECISION",
        "controlling_claim_text": "fixture weak claim 0.12345",
        "forbidden_claims": ["fixture forbidden claim"],
        "expansion_allowed": False,
        "systems": {
            "two_tank": {
                "coverage_0_05_margin": 0.12345,
                "coverage_0_10_margin": 0.23456,
                "effect_strength": "fixture_strength",
            },
            "cstr": {
                "coverage_0_05_margin": 0.34567,
                "coverage_0_10_margin": 0.45678,
                "effect_strength": "fixture_weak",
            },
        },
        "signal_roles": {
            "repair_amount": {"cstr_role": "diagnostic_only", "reason": "fixture repair reason"},
            "invariant_residual": {"cstr_role": "informative_refusal_signal", "reason": "fixture invariant reason"},
        },
        "allowed_next_action": "FIXTURE_NEXT_ACTION",
        "source_artifacts": {"fixture": "fixture/path.json"},
    }

    write_current_evidence_manifest_report(manifest, output)

    text = output.read_text(encoding="utf-8")
    assert "fixture weak claim 0.12345" in text
    assert "fixture repair reason" in text
    assert "0.456780" in text
