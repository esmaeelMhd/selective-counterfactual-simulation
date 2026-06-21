from __future__ import annotations

import json
from pathlib import Path

from scs.experiments.effect_audit import make_practical_utility_decision_gate


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _decision(tmp_path: Path, effect: str, forensics: str, event: str) -> str:
    result = make_practical_utility_decision_gate(
        _write(tmp_path / "effect.json", {"verdict": effect}),
        _write(tmp_path / "forensics.json", {"verdict": forensics}),
        _write(tmp_path / "event.json", {"verdict": event}),
        tmp_path / "gate.md",
    )
    return result["decision"]


def test_strong_fixture_allows_technical_report(tmp_path: Path) -> None:
    assert _decision(tmp_path, "STRONG_TWO_SYSTEM_EFFECT", "FALSE_ACCEPTS_EXPLAINED", "EVENT_SUPPORTS_CLAIM") == "WRITE_TECHNICAL_REPORT"


def test_weak_cstr_effect_narrows_claim(tmp_path: Path) -> None:
    assert _decision(tmp_path, "WEAK_TWO_SYSTEM_EFFECT", "INCONCLUSIVE", "EVENT_SUPPORTS_CLAIM") == "NARROW_TO_WEAK_LOW_COVERAGE_CLAIM"


def test_event_risk_failure_prevents_writing_report(tmp_path: Path) -> None:
    assert _decision(tmp_path, "WEAK_TWO_SYSTEM_EFFECT", "FALSE_ACCEPTS_EXPLAINED", "EVENT_WEAKENS_CLAIM") != "WRITE_TECHNICAL_REPORT"
