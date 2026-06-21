from __future__ import annotations

import pandas as pd

from scs.experiments.repair_signal_semantics import _controlled_case_rows, repair_validation_verdict


def test_controlled_repair_cases_cover_bounds_and_within_bound_errors() -> None:
    rows = _controlled_case_rows("cstr", dt=0.1, epsilon=1e-9, bad_threshold=0.15)
    frame = pd.DataFrame(rows)
    out_of_bounds = frame[frame["expected_repair_positive"]]
    wrong = frame[frame["case_id"].str.startswith("within_bounds_wrong")]
    assert not out_of_bounds.empty
    assert (out_of_bounds["repair_amount"] > 1e-9).all()
    assert not wrong.empty
    assert (wrong["repair_amount"] <= 1e-9).all()
    assert (wrong["rmse"] > 0.15).all()


def test_repair_implementation_bug_verdict_fires_on_zero_out_of_bounds_repair() -> None:
    fixture = pd.DataFrame(
        [
            {
                "system_id": "cstr",
                "case_id": "out_of_bounds_temperature",
                "has_bounds": True,
                "has_repair_operator": True,
                "expected_repair_positive": True,
                "repair_amount": 0.0,
                "rmse": 10.0,
            }
        ]
    )
    assert repair_validation_verdict(fixture, 1e-9, 0.15) == "REPAIR_IMPLEMENTATION_BUG"


def test_repair_irrelevant_verdict_fires_on_correct_bounds_but_wrong_dynamics() -> None:
    fixture = pd.DataFrame(
        [
            {
                "system_id": "cstr",
                "case_id": "out_of_bounds_temperature",
                "has_bounds": True,
                "has_repair_operator": True,
                "expected_repair_positive": True,
                "repair_amount": 2.0,
                "rmse": 10.0,
            },
            {
                "system_id": "cstr",
                "case_id": "within_bounds_wrong_temperature_trajectory",
                "has_bounds": True,
                "has_repair_operator": True,
                "expected_repair_positive": False,
                "repair_amount": 0.0,
                "rmse": 2.0,
            },
        ]
    )
    assert repair_validation_verdict(fixture, 1e-9, 0.15) == "REPAIR_CORRECT_BUT_CSTR_IRRELEVANT"
