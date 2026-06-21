from __future__ import annotations

from scs.experiments.repair_signal_semantics import repair_role_decision


def test_decision_gate_fix_repair_implementation_fixture() -> None:
    decision, action = repair_role_decision(
        {"verdict": "REPAIR_IMPLEMENTATION_BUG"},
        {"verdict": "INCONCLUSIVE"},
        {"verdict": "NO_REPAIR_NO_BENEFIT", "cstr_no_repair_mean_delta_margin_vs_full": 0.0},
        {"verdict": "NO_SEED_STABLE_BENEFIT"},
    )
    assert decision == "FIX_REPAIR_IMPLEMENTATION"
    assert action == "IMPLEMENT_REPAIR_BUG_FIX"


def test_decision_gate_system_specific_fixture() -> None:
    decision, action = repair_role_decision(
        {"verdict": "REPAIR_CORRECT_BUT_CSTR_IRRELEVANT"},
        {"verdict": "INVARIANT_DOMINATES_REPAIR"},
        {"verdict": "NO_REPAIR_IMPROVES_CSTR_WITHOUT_HURTING_TWOTANK", "cstr_no_repair_mean_delta_margin_vs_full": 0.03},
        {"verdict": "SEED_STABLE_NO_REPAIR_BENEFIT"},
    )
    assert decision == "MARK_REPAIR_SYSTEM_SPECIFIC"
    assert action == "IMPLEMENT_SYSTEM_SPECIFIC_SIGNAL_GATING"


def test_decision_gate_keep_universal_fixture() -> None:
    decision, action = repair_role_decision(
        {"verdict": "REPAIR_CORRECT_BUT_CSTR_IRRELEVANT"},
        {"verdict": "BOTH_USEFUL"},
        {"verdict": "NO_REPAIR_NO_BENEFIT", "cstr_no_repair_mean_delta_margin_vs_full": -0.01},
        {"verdict": "NO_SEED_STABLE_BENEFIT"},
    )
    assert decision == "KEEP_REPAIR_UNIVERSAL"
    assert action == "KEEP_CURRENT_WEAK_CLAIM"


def test_decision_gate_inconclusive_fixture() -> None:
    decision, action = repair_role_decision(
        {"verdict": "INCONCLUSIVE"},
        {"verdict": "INCONCLUSIVE"},
        {"verdict": "SIGNAL_SET_EFFECT_INCONCLUSIVE", "cstr_no_repair_mean_delta_margin_vs_full": -0.01},
        {"verdict": "CSTR_ONLY_UNSTABLE_BENEFIT"},
    )
    assert decision == "INCONCLUSIVE"
    assert action == "DO_NOT_EXPAND"
