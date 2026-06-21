from __future__ import annotations

from pathlib import Path

from scs.experiments.current_status import (
    write_claim_language_report,
    write_current_status_decision_report,
    write_readme_sync_report,
)


def test_current_status_decision_report_uses_supplied_values(tmp_path: Path) -> None:
    output = tmp_path / "decision.md"
    write_current_status_decision_report(
        {
            "inputs": {
                "preconditions": "FIXTURE_PRECONDITION",
                "manifest_status_id": "fixture_status",
                "signal_sync": "FIXTURE_SIGNAL",
                "readme_sync": "FIXTURE_README",
                "claim_language": "FIXTURE_CLAIMS",
            },
            "decision": "FIXTURE_DECISION",
            "allowed_next_action": "FIXTURE_ACTION",
            "forbidden_next_actions": ["fixture forbidden direction"],
            "allowed_claim": "fixture allowed claim",
            "forbidden_claims": ["fixture forbidden claim"],
        },
        output,
    )

    text = output.read_text(encoding="utf-8")
    assert "FIXTURE_DECISION" in text
    assert "fixture allowed claim" in text
    assert "fixture forbidden direction" in text


def test_readme_sync_and_claim_reports_use_supplied_values(tmp_path: Path) -> None:
    readme_report = tmp_path / "readme.md"
    claim_report = tmp_path / "claims.md"

    write_readme_sync_report(
        {"readme": "README_FIXTURE.md", "manifest_status_id": "fixture_manifest", "check_mode": True, "verdict": "FIXTURE_README_VERDICT"},
        readme_report,
    )
    write_claim_language_report(
        {
            "paths_scanned": ["fixture.md"],
            "risk_phrases": ["fixture risk phrase"],
            "violations": [{"path": "fixture.md", "line": 7, "phrase": "fixture risk phrase", "context_heading": "fixture"}],
            "allowed_negated_or_forbidden_mentions": [],
            "verdict": "FIXTURE_CLAIM_VERDICT",
        },
        claim_report,
    )

    assert "FIXTURE_README_VERDICT" in readme_report.read_text(encoding="utf-8")
    text = claim_report.read_text(encoding="utf-8")
    assert "fixture risk phrase" in text
    assert "FIXTURE_CLAIM_VERDICT" in text
