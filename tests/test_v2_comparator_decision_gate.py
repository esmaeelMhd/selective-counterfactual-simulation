from __future__ import annotations

from scs.experiments.v2_comparator import decision_from_statistics


def test_decision_gate_fixture_labels() -> None:
    cases = [
        (
            {"verdict": "CALIBRATED_FAILS_FAIR_DEPLOYABLE_BASELINE", "event_risk_worsening_count": 0},
            "CALIBRATED_FAILS_FAIR_BASELINE",
        ),
        (
            {"verdict": "CALIBRATED_BEATS_FAIR_DEPLOYABLE_BASELINE", "event_risk_worsening_count": 0},
            "CALIBRATED_BEATS_FAIR_BASELINE_ONLY",
        ),
        (
            {"verdict": "CALIBRATED_TARGET_DEPENDENT", "event_risk_worsening_count": 1},
            "CALIBRATED_TARGET_DEPENDENT",
        ),
        (
            {"verdict": "COMPARATOR_ENVELOPE_TOO_STRICT_BUT_METHOD_WEAK", "event_risk_worsening_count": 0},
            "COMPARATOR_TOO_STRICT_BUT_METHOD_STILL_WEAK",
        ),
        (
            {"verdict": "INVALID_COMPARATOR_STATISTICS", "event_risk_worsening_count": 0},
            "INVALID_COMPARATOR_ANALYSIS",
        ),
    ]
    for stats, expected in cases:
        decision, _ = decision_from_statistics(stats)
        assert decision == expected


def test_event_risk_failure_blocks_upgrade() -> None:
    decision, claim = decision_from_statistics(
        {"verdict": "CALIBRATED_BEATS_FAIR_DEPLOYABLE_BASELINE", "event_risk_worsening_count": 1}
    )
    assert decision == "CALIBRATED_TARGET_DEPENDENT"
    assert "event-risk" in claim
