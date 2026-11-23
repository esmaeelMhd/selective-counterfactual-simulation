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
    assert config["rmse_threshold"]["bsm1"] == pytest.approx(205.869)
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
