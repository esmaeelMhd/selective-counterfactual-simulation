from __future__ import annotations

from pathlib import Path

import pytest

from scs.experiments.registry import make_system
from scs.experiments.v2 import _make_v2_model
from scs.experiments.v3 import load_v3_config, protocol_is_frozen, run_v3_stage, stage_b_blockers
from scs.systems.bsm1 import confirmatory_window_capacity


def test_protocol_draft_is_not_frozen() -> None:
    assert protocol_is_frozen() is False


def test_stage_b_lists_the_open_decisions() -> None:
    config = load_v3_config("configs/v3/stage_b.yaml")
    blockers = stage_b_blockers(config)
    text = "\n".join(blockers)
    assert config["rmse_threshold"]["bsm1"] == pytest.approx(61.761)
    assert config["check_threshold_share"] is True
    assert "protocol is frozen" in text
    assert "refusal guard" in text
    assert "5% and 10%" in text
    assert "bad_rmse threshold is unset" not in text
    assert "overlapping" in text
    assert "control-only" in text


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
            "n_rehearsal_test": 4,
        },
    }


def test_bsm1_stage_dataset_builds_the_rehearsal_test_from_calibration_days(monkeypatch) -> None:
    pytest.importorskip("bsm2_python")
    from scs.systems import bsm1

    state = None

    def fake_states(dry, starts):
        nonlocal state
        if state is None:
            from tests.test_v3_bsm1_interface import _state_after_short_rollout

            state = _state_after_short_rollout()
        return {int(start): state for start in starts}

    monkeypatch.setattr(bsm1, "_cached_start_states", fake_states)
    dataset = bsm1.generate_bsm1_stage_dataset(_tiny_bsm1_config("B-rehearsal"), seed=1)
    assert set(dataset) == {"model_train", "judge_calibration_id", "judge_calibration_control", "judge_test_rehearsal"}
    assert dataset["judge_test_rehearsal"].n_trajectories == 4
    assert dataset["judge_test_rehearsal"].system_id == "bsm1"
    assert set(dataset["judge_test_rehearsal"].scenario_type) == {"rehearsal_nominal", "rehearsal_control_shift"}
    specs = bsm1._rehearsal_specs(4, 4, 2, 1)
    assert max(int(spec["start_minute"]) for spec in specs) + 6 <= bsm1.reserved_minutes()[0]
    # Without the rehearsal stage there is no test split at all.
    plain = bsm1.generate_bsm1_stage_dataset(_tiny_bsm1_config("A"), seed=1)
    assert "judge_test_rehearsal" not in plain


def test_confirmatory_stage_has_no_dataset_generator_yet() -> None:
    from scs.systems.bsm1 import generate_bsm1_stage_dataset

    with pytest.raises(RuntimeError, match="not generated before the freeze decisions"):
        generate_bsm1_stage_dataset(_tiny_bsm1_config("B"), seed=0)


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
