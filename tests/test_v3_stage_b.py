from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scs.experiments.registry import make_system
from scs.experiments.v2 import _make_v2_model
from scs.experiments.v3 import load_v3_config, protocol_is_frozen, run_v3_stage, stage_b_blockers
from scs.systems.bsm1 import confirmatory_window_capacity


def test_protocol_draft_is_not_frozen() -> None:
    assert protocol_is_frozen() is False


def test_only_the_freeze_itself_blocks_stage_b_once_the_decisions_are_drafted(monkeypatch) -> None:
    from scs.experiments import v3

    monkeypatch.setattr(v3, "_tracked_working_tree_dirty", lambda: False)
    config = load_v3_config("configs/v3/stage_b.yaml")
    blockers = stage_b_blockers(config)
    assert config["rmse_metric"] == "standardized_state"
    assert config["rmse_threshold"]["bsm1"] == pytest.approx(0.4691)
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
    assert nominal.measured_history == pytest.approx(shifted.measured_history)
    assert nominal.states[:, 0, :] == pytest.approx(shifted.states[:, 0, :])
    assert not np.allclose(nominal.states[:, 1:, :], shifted.states[:, 1:, :])
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


def test_threshold_share_check_runs_before_generating_confirmatory_test_splits(
    fast_start_states, tmp_path: Path, monkeypatch
) -> None:
    import yaml

    bsm1 = fast_start_states
    called = {"confirmatory": False}

    def fail_if_called(**kwargs):
        called["confirmatory"] = True
        raise AssertionError("confirmatory generator should not run before threshold_share_check")

    monkeypatch.setattr(bsm1, "generate_bsm1_confirmatory_dataset", fail_if_called)
    config = {
        "stage": "B-rehearsal",
        "label": "rehearsal",
        "protocol_frozen": False,
        "event_config": "configs/v3/bsm1_events.yaml",
        "check_threshold_share": True,
        "systems": ["bsm1"],
        "models": ["hold_last", "linear_narx", "mlp_state_space"],
        "learned_models": ["mlp_state_space"],
        "eligibility_split": "judge_calibration_id",
        "eligibility_baselines": ["hold_last", "linear_narx"],
        "primary_coverages": [0.05, 0.10],
        "badness_targets": ["bad_rmse"],
        "rmse_threshold": {"bsm1": 1e9},
        "rmse_metric": "standardized_state",
        "event_guards": {"event_guard_no_invariant": ["risk_support_only", "risk_uncertainty_only", "risk_disagreement_only"]},
        "seeds": [0],
        "bootstrap_samples": 10,
        "bootstrap_seed": 1,
        "interval_width_limit": 0.10,
        "min_test_scenarios": 4,
        "degenerate_share": 0.01,
        "data": {
            "horizon": 4,
            "history_minutes": 2,
            "dt": 1.0 / 1440.0,
            "n_model_train": 6,
            "n_calibration_id": 3,
            "n_calibration_ood": 3,
            "n_test_per_split": 2,
            "uncertainty_samples": 5,
        },
        "model_runtime": {"ensemble_members": 5, "mlp_max_iter": 10},
    }
    # Force mlp_state_space eligible so threshold_share_check checks it.
    from scs.experiments import v3

    monkeypatch.setattr(
        v3,
        "eligibility_table",
        lambda frame, cfg: __import__("pandas").DataFrame(
            [{"system_id": "bsm1", "model_id": "mlp_state_space", "id_rmse": 0.1, "baseline_hold_last_id_rmse": 1.0, "baseline_linear_narx_id_rmse": 0.5, "eligible": True}]
        ),
    )
    cfg_path = tmp_path / "rehearsal_bad_thresh.yaml"
    cfg_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(RuntimeError, match="marks 0.0% of bsm1"):
        run_v3_stage(cfg_path, tmp_path / "out")
    assert called["confirmatory"] is False


def test_stage_run_writes_git_commit_and_protocol_hash_into_results_and_report(
    fast_start_states, tmp_path: Path
) -> None:
    import pandas as pd
    import yaml

    config = {
        "stage": "B-rehearsal",
        "label": "rehearsal",
        "protocol_frozen": False,
        "event_config": "configs/v3/bsm1_events.yaml",
        "check_threshold_share": False,
        "systems": ["bsm1"],
        "models": ["hold_last", "linear_narx"],
        "learned_models": [],
        "eligibility_split": "judge_calibration_id",
        "eligibility_baselines": ["hold_last", "linear_narx"],
        "primary_coverages": [0.05, 0.10],
        "badness_targets": ["bad_rmse"],
        "rmse_threshold": {"bsm1": 0.5},
        "rmse_metric": "standardized_state",
        "event_guards": {"event_guard_no_invariant": ["risk_support_only", "risk_uncertainty_only", "risk_disagreement_only"]},
        "seeds": [0],
        "bootstrap_samples": 10,
        "bootstrap_seed": 1,
        "interval_width_limit": 0.10,
        "min_test_scenarios": 4,
        "data": {
            "horizon": 4,
            "history_minutes": 2,
            "dt": 1.0 / 1440.0,
            "n_model_train": 4,
            "n_calibration_id": 2,
            "n_calibration_ood": 2,
            "n_test_per_split": 2,
            "uncertainty_samples": 5,
            "cache_dir": str(tmp_path / "cache"),
        },
        "model_runtime": {"ensemble_members": 5},
    }
    cfg_path = tmp_path / "rehearsal.yaml"
    cfg_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    out_dir = tmp_path / "out"
    run_v3_stage(cfg_path, out_dir)
    table = pd.read_csv(out_dir / "stage_b_rehearsal_results.csv")
    report = (out_dir / "stage_b_rehearsal_report.md").read_text(encoding="utf-8")
    assert "git_commit" in table.columns and "protocol_sha256" in table.columns
    assert str(table["protocol_sha256"].iloc[0]) in report
    assert (tmp_path / "cache" / "seed_0" / "pretest" / "model_train.npz").is_file()
    assert (tmp_path / "cache" / "seed_0" / "test" / "judge_test_dry_holdout.npz").is_file()


def test_stage_b_refuses_dirty_tracked_working_tree_even_when_frozen(monkeypatch) -> None:
    from scs.experiments import v3

    config = {**load_v3_config("configs/v3/stage_b.yaml"), "protocol_frozen": True}
    monkeypatch.setattr(v3, "protocol_is_frozen", lambda *args, **kwargs: True)
    monkeypatch.setattr(v3, "_tracked_working_tree_dirty", lambda: True)
    dirty_blockers = v3.stage_b_blockers(config)
    assert any("uncommitted" in item.lower() for item in dirty_blockers)
    monkeypatch.setattr(v3, "_tracked_working_tree_dirty", lambda: False)
    assert v3.stage_b_blockers(config) == []


def test_stage_b_blockers_validates_decision_values_and_test_scenario_count(monkeypatch) -> None:
    from scs.experiments import v3

    config = {**load_v3_config("configs/v3/stage_b.yaml"), "protocol_frozen": True}
    monkeypatch.setattr(v3, "protocol_is_frozen", lambda *args, **kwargs: True)
    monkeypatch.setattr(v3, "_tracked_working_tree_dirty", lambda: False)
    assert any("coverage_rule" in item for item in v3.stage_b_blockers({**config, "coverage_rule": "bogus"}))
    assert any("control_split_influent" in item for item in v3.stage_b_blockers({**config, "control_split_influent": "other"}))
    assert any("overlapping" in item.lower() for item in v3.stage_b_blockers({**config, "overlapping_test_windows": False}))
    small_test = {**config, "data": {**config["data"], "n_test_per_split": 100}}
    assert any("2000" in item for item in v3.stage_b_blockers(small_test))


def test_bsm1_cache_checks_freeze_before_reading_and_invalidates_on_config_change(
    fast_start_states, tmp_path: Path
) -> None:
    bsm1 = fast_start_states
    cache_root = str(tmp_path / "cache")
    rehearsal_cfg = {
        "stage": "B-rehearsal",
        "data": {
            "horizon": 4,
            "history_minutes": 2,
            "n_model_train": 3,
            "n_calibration_id": 2,
            "n_calibration_ood": 2,
            "n_test_per_split": 2,
            "cache_dir": cache_root,
        },
    }
    first = bsm1.generate_bsm1_test_stage_dataset(rehearsal_cfg, seed=0)
    assert first["judge_test_dry_holdout"].states.shape[1] == 5

    # Reusing the same cache_dir with stage="B" while unfrozen MUST refuse before reading cache.
    stage_b_cfg = {**rehearsal_cfg, "stage": "B"}
    with pytest.raises(RuntimeError, match="only after the freeze"):
        bsm1.generate_bsm1_test_stage_dataset(stage_b_cfg, seed=0)

    # Changing horizon on the same cache_dir MUST invalidate stale cache and rebuild.
    longer_cfg = {**rehearsal_cfg, "data": {**rehearsal_cfg["data"], "horizon": 6}}
    rebuilt = bsm1.generate_bsm1_test_stage_dataset(longer_cfg, seed=0)
    assert rebuilt["judge_test_dry_holdout"].states.shape[1] == 7


def test_mixed_system_config_does_not_attach_bsm1_projection_to_ode_models() -> None:
    config = {"systems": ["two_tank", "bsm1"], "model_runtime": {"early_stopping": True}}
    ode_model = _make_v2_model("lstm_world", 0, config, system_id="two_tank")
    bsm1_model = _make_v2_model("lstm_world", 0, config, system_id="bsm1")
    assert ode_model.projection_factory is None
    assert bsm1_model.projection_factory is not None
