from __future__ import annotations

from scs.experiments.current_status import current_status_decision


def _valid_inputs() -> tuple[dict, dict, dict, dict, dict]:
    return (
        {"verdict": "READY_FOR_STATUS_SYNC"},
        {"status_id": "current_evidence_status_v1", "expansion_allowed": False, "allowed_next_action": "UPDATE_SIGNAL_SEMANTICS_ONLY"},
        {"verdict": "SIGNAL_SEMANTICS_SYNCED"},
        {"verdict": "README_SYNCED"},
        {"verdict": "CLAIM_LANGUAGE_OK"},
    )


def test_current_status_decision_gate_accepts_only_synced_inputs() -> None:
    assert current_status_decision(*_valid_inputs()) == "CURRENT_STATUS_SYNCED"


def test_current_status_decision_gate_blocks_expansion() -> None:
    preconditions, manifest, signal_sync, readme_sync, claim_language = _valid_inputs()
    manifest["expansion_allowed"] = True

    assert current_status_decision(preconditions, manifest, signal_sync, readme_sync, claim_language) == "CURRENT_STATUS_STALE_OR_UNSAFE"
