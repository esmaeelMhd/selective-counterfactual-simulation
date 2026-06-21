from __future__ import annotations

from scs.experiments.cstr_weakness import _model_scenario_verdict


def test_model_concentration_rule() -> None:
    summary = {"accepted_false_accept_count": 10, "top_model_share_of_false_accepts": 0.7, "top_scenario_share_of_false_accepts": 0.2}
    assert _model_scenario_verdict(summary) == "MODEL_SPECIFIC_CSTR_FAILURE"


def test_scenario_concentration_rule() -> None:
    summary = {"accepted_false_accept_count": 10, "top_model_share_of_false_accepts": 0.4, "top_scenario_share_of_false_accepts": 0.7}
    assert _model_scenario_verdict(summary) == "SCENARIO_SPECIFIC_CSTR_FAILURE"


def test_diffuse_failure_rule() -> None:
    summary = {"accepted_false_accept_count": 10, "top_model_share_of_false_accepts": 0.4, "top_scenario_share_of_false_accepts": 0.4}
    assert _model_scenario_verdict(summary) == "DIFFUSE_CSTR_FAILURE"
