from __future__ import annotations

import json
from pathlib import Path

from scs.experiments.cstr_replication import make_multi_system_calibrated_decision_gate


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _gate(tmp_path: Path, cstr_seed: str = "ROBUST_LOW_COVERAGE", cstr_single: str = "SUPPORTED_LOW_COVERAGE", cstr_stress: str = "ROBUST_LOW_COVERAGE_ONLY", cstr_sanity: str = "VALID_CSTR_BENCHMARK", leakage: bool = False) -> dict:
    return make_multi_system_calibrated_decision_gate(
        twotank_single=_write(tmp_path / "twotank_single.json", {"verdict": "SUPPORTED_LOW_COVERAGE", "leakage_detected": False}),
        twotank_seed=_write(tmp_path / "twotank_seed.json", {"verdict": "ROBUST_LOW_COVERAGE", "leakage_detected": False}),
        twotank_stress=_write(tmp_path / "twotank_stress.json", {"verdict": "ROBUST_LOW_COVERAGE_ONLY", "leakage_detected": False}),
        cstr_sanity=_write(tmp_path / "cstr_sanity.json", {"verdict": cstr_sanity}),
        cstr_single=_write(tmp_path / "cstr_single.json", {"verdict": cstr_single, "leakage_detected": leakage}),
        cstr_seed=_write(tmp_path / "cstr_seed.json", {"verdict": cstr_seed, "leakage_detected": False}),
        cstr_stress=_write(tmp_path / "cstr_stress.json", {"verdict": cstr_stress, "leakage_detected": False}),
        output=tmp_path / "gate.md",
    )


def test_multi_system_gate_supported_case(tmp_path: Path) -> None:
    assert _gate(tmp_path)["decision"] == "TWO_SYSTEM_LOW_COVERAGE_SUPPORTED"


def test_multi_system_gate_cstr_failure_modes(tmp_path: Path) -> None:
    assert _gate(tmp_path / "only", cstr_seed="NO_ROBUST_IMPROVEMENT")["decision"] == "TWOTANK_ONLY_SUPPORTED"
    assert _gate(tmp_path / "mixed", cstr_seed="UNSTABLE")["decision"] == "MIXED_SYSTEM_EVIDENCE"
    assert _gate(tmp_path / "nogeneral", cstr_single="NO_IMPROVEMENT_OVER_SINGLE_SIGNAL", cstr_seed="NO_ROBUST_IMPROVEMENT", cstr_stress="NO_STABLE_REGION")["decision"] == "TWOTANK_ONLY_SUPPORTED"


def test_multi_system_gate_invalid_cases(tmp_path: Path) -> None:
    assert _gate(tmp_path / "leak", leakage=True)["decision"] == "INVALID_DUE_TO_LEAKAGE"
    assert _gate(tmp_path / "invalid", cstr_sanity="INVALID_CSTR_BENCHMARK")["decision"] == "INVALID_CSTR_BENCHMARK"
