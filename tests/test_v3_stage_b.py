from __future__ import annotations

from pathlib import Path

import pytest

from scs.experiments.registry import make_system
from scs.experiments.v2 import _make_v2_model
from scs.experiments.v3 import load_v3_config, protocol_is_frozen, run_v3_stage, stage_b_blockers
from scs.systems.bsm1 import confirmatory_window_capacity


def test_protocol_draft_is_not_frozen() -> None:
    assert protocol_is_frozen() is False


def test_only_the_freeze_itself_blocks_stage_b_once_the_decisions_are_drafted() -> None:
    config = load_v3_config("configs/v3/stage_b.yaml")
    blockers = stage_b_blockers(config)
    assert config["rmse_threshold"]["bsm1"] == pytest.approx(61.761)
    assert config["check_threshold_share"] is True
    assert config["coverage_rule"] == "both"
    assert config["event_guard_on_bsm1"] == "event_guard_no_invariant"
    assert config["overlapping_test_windows"] is True
    assert config["data"]["n_test_per_split"] * 4 >= config["min_test_scenarios"]
    assert len(config["seeds"]) == 10
    assert len(blockers) == 2
    text = "\n".join(blockers)
    assert "is still a draft" in text
    assert "protocol_frozen" in text


def test_each_missing_decision_blocks_stage_b_again() -> None:
    config = load_v3_config("configs/v3/stage_b.yaml")
    for key, phrase in (
        ("coverage_rule", "5% and 10%"),
        ("event_guard_on_bsm1", "refusal guard is unset"),
        ("control_split_influent", "control-only"),
    ):
        text = "\n".join(stage_b_blockers({**config, key: None}))
        assert phrase in text, key
    assert "overlapping" in "\n".join(stage_b_blockers({**config, "overlapping_test_windows": None}))


def test_stage_b_run_stops_before_writing(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="not run before the freeze"):
        run_v3_stage("configs/v3/stage_b.yaml", tmp_path / "stage_b")
    assert not (tmp_path / "stage_b").exists()


def test_reserved_weather_holds_few_nonoverlapping_windows() -> None:
    capacity = confirmatory_window_capacity()
    assert capacity["span_minutes"] == 495
    assert capacity["dry_nonoverlapping"] == 4
    assert capacity["rain_nonoverlapping"] == 8


def test_bsm1_system_refuses_the_full_state_signals() -> None:
    system = make_system("bsm1")
    with pytest.raises(RuntimeError, match="invariant_residual"):
        system.invariant_residual(None, None, None, 1.0)
    with pytest.raises(RuntimeError, match="repair_amount"):
        system.repair_amount(None, None)


def test_stage_b_budget_reaches_the_learned_models() -> None:
    config = load_v3_config("configs/v3/stage_b.yaml")
    lstm = _make_v2_model("lstm_world", 0, config)
    assert lstm.early_stopping is True
    assert lstm.validation_gap == 18
    assert lstm.epochs == 120
    assert lstm.projection_factory is not None
    recurrent = _make_v2_model("recurrent_world", 0, config)
    assert recurrent.early_stopping is True
    assert recurrent.projection_factory is not None


def test_ode_models_keep_the_short_budget() -> None:
    config = {"systems": ["two_tank"], "model_runtime": {}}
    model = _make_v2_model("lstm_world", 0, config)
    assert model.early_stopping is False
    assert model.epochs == 30
    assert model.projection_factory is None


def _tiny_bsm1_config(stage: str) -> dict:
    return {
        "stage": stage,
        "data": {
            "horizon": 4,
            "history_minutes": 2,
            "n_model_train": 3,
            "n_calibration_id": 2,
            "n_calibration_ood": 2,
            "n_test_per_split": 3,
        },
    }


@pytest.fixture
def fast_start_states(monkeypatch):
    """Every window starts from one short-run plant state, so these tests do not run the 14-day warm-up."""
    pytest.importorskip("bsm2_python")
    from scs.systems import bsm1
    from tests.test_v3_bsm1_interface import _state_after_short_rollout

    state = _state_after_short_rollout()
    monkeypatch.setattr(bsm1, "_cached_start_states", lambda dry, starts: {int(start): state for start in starts})
    monkeypatch.setattr(bsm1, "_cached_rain_start_states", lambda dry, rain, starts: {int(start): state for start in starts})
    return bsm1


def test_rehearsal_builds_the_four_test_splits_from_days_that_are_not_reserved(fast_start_states) -> None:
    bsm1 = fast_start_states
    dataset = bsm1.generate_bsm1_stage_dataset(_tiny_bsm1_config("B-rehearsal"), seed=1)
    tests = {name: batch for name, batch in dataset.items() if name.startswith("judge_test")}
    assert set(tests) == set(bsm1.CONFIRMATORY_SPLITS)
    assert all(batch.n_trajectories == 3 and batch.system_id == "bsm1" for batch in tests.values())
    assert set(dataset) - set(tests) == {"model_train", "judge_calibration_id", "judge_calibration_control"}
    for weather, (first, end) in bsm1.confirmatory_blocks(rehearsal=True).items():
        assert not bsm1._reads_reserved(weather, first, end)
    # Rain flows are in m3/d like the dry flows, not in thousands.
    assert float(tests["judge_test_rain"].disturbances[..., bsm1.FLOW_INDEX].min()) > 1000.0


def test_control_windows_pair_with_the_nominal_windows_and_differ_in_aeration_only(fast_start_states) -> None:
    bsm1 = fast_start_states
    tests = bsm1.generate_bsm1_confirmatory_dataset(n_per_split=3, horizon=4, history=2, seed=2, rehearsal=True)
    nominal, shifted = tests["judge_test_dry_holdout"], tests["judge_test_dry_control"]
    assert nominal.disturbances == pytest.approx(shifted.disturbances)
    assert nominal.measured_history.shape == shifted.measured_history.shape
    assert (shifted.actions[:, 0, 2] > nominal.actions[:, 0, 2].max()).all()
    low, high = bsm1.CONTROL_SCALE
    ratio = shifted.actions[:, 0, 2] / bsm1.NOMINAL_KLA[2]
    assert ((ratio >= low) & (ratio <= high)).all()
    wider = bsm1.generate_bsm1_confirmatory_dataset(
        n_per_split=3, horizon=4, history=2, seed=2, control_scale=(1.9, 2.2), rehearsal=True
    )["judge_test_dry_control"]
    assert (wider.actions[:, 0, 2] / bsm1.NOMINAL_KLA[2] >= 1.9).all()


def test_the_real_confirmatory_blocks_are_refused_until_the_protocol_is_frozen(fast_start_states, monkeypatch) -> None:
    bsm1 = fast_start_states
    with pytest.raises(RuntimeError, match="only after the freeze"):
        bsm1.generate_bsm1_stage_dataset(_tiny_bsm1_config("B"), seed=0)
    monkeypatch.setattr("scs.experiments.v3.protocol_is_frozen", lambda *args, **kwargs: True)
    dataset = bsm1.generate_bsm1_stage_dataset(_tiny_bsm1_config("B"), seed=0)
    reserved = bsm1.reserved_minutes()
    event = bsm1._rain_event_minutes()
    assert bsm1.confirmatory_blocks()["dry"] == reserved and bsm1.confirmatory_blocks()["rain"] == event
    assert "judge_test_rain_control" in dataset and dataset["judge_test_dry_holdout"].n_trajectories == 3


def test_a_stage_without_test_splits_adds_none(fast_start_states) -> None:
    dataset = fast_start_states.generate_bsm1_stage_dataset(_tiny_bsm1_config("A"), seed=1)
    assert not any(name.startswith("judge_test") for name in dataset)


def test_threshold_check_refuses_a_threshold_that_marks_nothing_bad() -> None:
    import pandas as pd

    from scs.experiments.v3 import threshold_share_check

    frame = pd.DataFrame(
        {
            "system_id": ["bsm1"] * 6,
            "role": ["judge_calibration"] * 6,
            "model_id": ["good"] * 6,
            "rmse": [5.0, 6.0, 7.0, 40.0, 50.0, 60.0],
        }
    )
    eligibility = pd.DataFrame([{"system_id": "bsm1", "model_id": "good", "eligible": True}])
    base = {"systems": ["bsm1"], "degenerate_share": 0.01}
    threshold_share_check(frame, eligibility, {**base, "rmse_threshold": {"bsm1": 20.0}})
    with pytest.raises(RuntimeError, match="marks 0.0% of bsm1"):
        threshold_share_check(frame, eligibility, {**base, "rmse_threshold": {"bsm1": 205.869}})
    with pytest.raises(RuntimeError, match="100.0%"):
        threshold_share_check(frame, eligibility, {**base, "rmse_threshold": {"bsm1": 1.0}})
    # Rows of models that are not eligible, or of test scenarios, are not read.
    other = pd.DataFrame([{"system_id": "bsm1", "model_id": "good", "eligible": False}])
    threshold_share_check(frame, other, {**base, "rmse_threshold": {"bsm1": 205.869}})


def test_bsm1_guard_must_be_defined_and_free_of_full_state_signals() -> None:
    from scs.experiments.v3 import stage_b_blockers

    config = load_v3_config("configs/v3/stage_b.yaml")
    assert not [item for item in stage_b_blockers(config) if "refusal guard" in item]
    unknown = {**config, "event_guard_on_bsm1": "not_a_guard"}
    assert any("not one of the guards" in item for item in stage_b_blockers(unknown))
    full_state = {**config, "event_guards": {"v2": ["risk_invariant_only", "risk_support_only"]}, "event_guard_on_bsm1": "v2"}
    assert any("does not have" in item for item in stage_b_blockers(full_state))
