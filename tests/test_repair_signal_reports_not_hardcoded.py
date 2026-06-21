from __future__ import annotations

from pathlib import Path

import pandas as pd

from scs.experiments.repair_signal_semantics import write_repair_validation_report, write_signal_set_ablation_report


def test_repair_validation_report_uses_supplied_numeric_values(tmp_path: Path) -> None:
    cases = pd.DataFrame(
        [
            {
                "system_id": "cstr",
                "case_id": "within_bounds_wrong_temperature_trajectory",
                "raw_state_violates_bounds": False,
                "repair_amount": 0.123456,
                "expected_repair_positive": False,
                "case_passed": True,
            }
        ]
    )
    summary = {
        "cstr_repair_status": "fixture cstr",
        "two_tank_repair_status": "fixture tank",
        "cstr_repair_semantic_status": "fixture 0.123456",
        "verdict": "REPAIR_CORRECT_BUT_CSTR_IRRELEVANT",
    }
    output = tmp_path / "repair.md"
    write_repair_validation_report(cases, summary, output)
    text = output.read_text(encoding="utf-8")
    assert "0.123456" in text
    assert "fixture cstr" in text


def test_signal_set_ablation_report_changes_with_fixture_values(tmp_path: Path) -> None:
    low = pd.DataFrame(
        [
            {
                "system_id": "cstr",
                "signal_set_id": "no_repair",
                "coverage": 0.05,
                "baseline_far": 0.8,
                "calibrated_far": 0.654321,
                "margin": 0.145679,
                "selected_judge": "rank_normalized_linear",
                "selected_signal_if_any": "invariant_residual",
                "leakage_detected": False,
            }
        ]
    )
    diff = pd.DataFrame([{"system_id": "cstr", "signal_set_id": "no_repair", "coverage": 0.05, "delta_margin_vs_full": 0.654321}])
    summary = {
        "cstr_no_repair_mean_delta_margin_vs_full": 0.654321,
        "two_tank_no_repair_mean_delta_margin_vs_full": 0.0,
        "verdict": "NO_REPAIR_IMPROVES_CSTR_WITHOUT_HURTING_TWOTANK",
    }
    output = tmp_path / "ablation.md"
    write_signal_set_ablation_report(low, diff, summary, output)
    assert "0.654321" in output.read_text(encoding="utf-8")
