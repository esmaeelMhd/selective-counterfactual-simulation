from __future__ import annotations

from pathlib import Path

import pytest

from scs.experiments.cstr_weakness import _final_diagnosis, make_cstr_weakness_diagnosis


def _summaries(repair: str = "REPAIR_SIGNAL_BLIND_SPOT") -> dict:
    return {
        "repair_signal": {"verdict": repair, "fraction_accepted_false_accepts_with_low_repair": 1.0},
        "statewise_error": {"verdict": "BOTH_STATES"},
        "signal_overlap": {"verdict": "SIGNAL_BLIND_SPOT"},
        "model_scenario_failure": {"verdict": "DIFFUSE_CSTR_FAILURE"},
        "rmse_target": {"verdict": "THRESHOLD_ROBUST_WEAK_EFFECT"},
    }


def test_final_diagnosis_rule_hierarchy_prefers_repair_blindspot() -> None:
    diagnosis, action = _final_diagnosis(_summaries())
    assert diagnosis == "REPAIR_SIGNAL_BLIND_SPOT"
    assert action == "FIX_REPAIR_SIGNAL"


def test_diffuse_signal_blindspot_blocks_expansion() -> None:
    diagnosis, action = _final_diagnosis(_summaries(repair="INCONCLUSIVE"))
    assert diagnosis == "DIFFUSE_SIGNAL_BLIND_SPOT"
    assert action == "DO_NOT_EXPAND"


def test_missing_input_summaries_cause_failure(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        make_cstr_weakness_diagnosis(tmp_path, tmp_path / "report.md")
