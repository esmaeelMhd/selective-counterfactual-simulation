from __future__ import annotations

from pathlib import Path

from scs.experiments.effect_audit import verify_effect_size_audit_preconditions


def test_effect_audit_scripts_do_not_add_expansion_surface() -> None:
    scripts = [
        Path("scripts/verify_effect_size_audit_preconditions.py"),
        Path("scripts/analyze_effect_size_uncertainty.py"),
        Path("scripts/analyze_accepted_false_accepts.py"),
        Path("scripts/analyze_event_risk.py"),
        Path("scripts/make_practical_utility_decision_gate.py"),
    ]
    forbidden = ["RSSM", "heat_exchanger", "FastAPI", "frontend", "dashboard"]
    for script in scripts:
        text = script.read_text(encoding="utf-8")
        assert not [token for token in forbidden if token in text]


def test_effect_audit_precondition_does_not_mutate_protocol_lock(tmp_path: Path) -> None:
    protocol = Path("docs/calibrated_protocol_lock_v1.md")
    before = protocol.read_text(encoding="utf-8")
    verify_effect_size_audit_preconditions("configs/audits/effect_size_audit.yaml", tmp_path / "preconditions")
    after = protocol.read_text(encoding="utf-8")
    assert before == after
