from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scs.experiments.v3 import eligibility_table, result_rows_for_frame
from scs.models.hold_last import HoldLastModel
from scs.signals import available_signals
from scs.systems.base import TrajectoryBatch


def _batch() -> TrajectoryBatch:
    return TrajectoryBatch(
        states=np.zeros((2, 3, 1)),
        actions=np.zeros((2, 2, 1)),
        disturbances=np.zeros((2, 2, 1)),
        scenario_type=["a", "b"],
        split="model_train",
        system_id="two_tank",
    )


def test_open_loop_flows_follow_the_influent() -> None:
    from scs.systems.bsm1 import FLOW_INDEX, REACTOR_WIDTH, apply_hydraulic_flows

    disturbances = np.zeros((2, 21))
    disturbances[:, FLOW_INDEX] = [18000.0, 20000.0]
    states = np.zeros((2, 136))
    updated = apply_hydraulic_flows(states, disturbances)
    reactor = disturbances[:, FLOW_INDEX] + 3.0 * 18446.0 + 18446.0
    effluent = disturbances[:, FLOW_INDEX] - 385.0
    for block in range(5):
        assert np.allclose(updated[:, FLOW_INDEX + REACTOR_WIDTH * block], reactor)
    assert np.allclose(updated[:, FLOW_INDEX + REACTOR_WIDTH * 5], effluent)


def test_saved_dry_flows_match_the_hydraulic_rule() -> None:
    from pathlib import Path

    from scs.data.schemas import load_batch
    from scs.systems.bsm1 import apply_hydraulic_flows

    path = Path("results/v3/bsm1_pretest/model_train.npz")
    if not path.exists():
        pytest.skip("dry pretest set is not present")
    batch = load_batch(path)
    projected = apply_hydraulic_flows(batch.states[:2, 1:], batch.disturbances[:2])
    assert np.allclose(projected[..., 14::21], batch.states[:2, 1:, 14::21])


def test_hydraulic_rollout_replaces_held_flow() -> None:
    from scs.systems.bsm1 import rollout_with_known_hydraulics

    model = HoldLastModel()
    model.fit(_batch())
    initial = np.zeros((1, 136))
    actions = np.zeros((1, 2, 5))
    disturbances = np.zeros((1, 2, 21))
    disturbances[:, :, 14] = 19000.0
    predicted = rollout_with_known_hydraulics(model, initial, actions, disturbances)
    assert np.allclose(predicted[0, 1, 14], 19000.0 + 4.0 * 18446.0)
    assert np.allclose(predicted[0, 2, 119], 19000.0 - 385.0)


def test_old_trajectory_batch_still_builds() -> None:
    batch = _batch()
    assert batch.measured_history is None
    assert batch.n_trajectories == 2


def test_measured_history_shape_is_checked() -> None:
    with pytest.raises(ValueError, match="measured_history"):
        TrajectoryBatch(
            states=np.zeros((1, 2, 1)),
            actions=np.zeros((1, 1, 1)),
            disturbances=np.zeros((1, 1, 1)),
            scenario_type=["a"],
            split="model_train",
            system_id="two_tank",
            measured_history=np.zeros((1, 0, 2)),
        )


def test_predict_rollout_accepts_history_and_keeps_the_old_call() -> None:
    model = HoldLastModel()
    initial = np.array([1.0, 2.0])
    actions = np.zeros((3, 1))
    disturbances = np.zeros((3, 1))
    history = np.ones((2, 5))
    without = model.predict_rollout(initial, actions, disturbances)
    with_history = model.predict_rollout(initial, actions, disturbances, history=history)
    assert with_history == pytest.approx(without)
    assert with_history.shape == (4, 2)


def test_bsm1_drops_full_state_signals() -> None:
    assert "invariant_residual" not in available_signals("bsm1")
    assert "repair_amount" not in available_signals("bsm1")
    assert "support_distance" in available_signals("bsm1")
    assert available_signals("two_tank") >= available_signals("bsm1")


def test_missing_signal_still_scores_and_the_guard_is_refused() -> None:
    rows = []
    for role, bad in (("judge_calibration", True), ("judge_calibration", False), ("judge_test", True), ("judge_test", False)):
        for index in range(3):
            rows.append(
                {
                    "system_id": "bsm1",
                    "seed": 0,
                    "split": "judge_test_id" if role == "judge_test" else "judge_calibration_id",
                    "role": role,
                    "model_id": "hold_last",
                    "rmse": 0.2 if bad else 0.01,
                    "event_mismatch": bad,
                    "support_distance": 1.0 if bad else 0.1,
                    "uncertainty_score": 0.5 if bad else 0.2,
                    "disagreement_score": 0.4 if bad else 0.1,
                    "invariant_residual": np.nan,
                    "repair_amount": np.nan,
                }
            )
    frame = pd.DataFrame(rows)
    config = {
        "learned_models": [],
        "eligibility_baselines": ["hold_last"],
        "primary_coverages": [0.5],
        "badness_targets": ["bad_rmse"],
        "rmse_threshold": 0.05,
        "event_guard_inputs": ["risk_invariant_only", "risk_disagreement_only", "risk_support_only"],
    }
    eligibility = eligibility_table(frame, config)
    results = result_rows_for_frame(frame, config, eligibility)
    judges = set(results["judge_id"])
    assert "invariant_only" not in judges
    assert "repair_only" not in judges
    assert "calibration_selected_candidate_ranker" in judges
    # The guard needs all three of its inputs. With the invariant signal missing it is not
    # computed on the two that remain, because that would be a different guard.
    assert "event_guard" not in judges
    assert np.isfinite(results["false_accept_rate"]).all()


def test_rain_storm_and_confirmatory_test_are_refused() -> None:
    from scs.systems.bsm1 import assert_split_allowed

    for split in ("judge_test_id", "judge_test_rain", "judge_test_storm", "judge_test_control", "rain", "storm"):
        with pytest.raises(RuntimeError):
            assert_split_allowed(split)


def test_pretest_splits_use_disjoint_dry_minutes_and_skip_the_reserved_tail() -> None:
    from scs.systems.bsm1 import (
        CONTROL_SCALE,
        DRY_MINUTES,
        HISTORY_MINUTES,
        HORIZON_MINUTES,
        NOMINAL_SCALE,
        pretest_specs,
        reserved_minutes,
        split_minutes,
    )

    span = HISTORY_MINUTES + HORIZON_MINUTES
    specs = pretest_specs(seed=0)
    assert {spec["split"] for spec in specs} == {"model_train", "judge_calibration_id", "judge_calibration_control"}
    reserved_first, reserved_end = reserved_minutes()
    assert reserved_end == DRY_MINUTES == 7 * 1440
    assert reserved_end - reserved_first == int(1.5 * 1440)
    used: dict[str, set[int]] = {}
    for spec in specs:
        first, end = split_minutes(str(spec["split"]))
        start = int(spec["start_minute"])
        assert first <= start and start + span <= end
        assert start + span <= reserved_first
        used.setdefault(str(spec["split"]), set()).update(range(start, start + span))
    names = sorted(used)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            assert not used[left] & used[right], (left, right)
    nominal = [spec for spec in specs if spec["split"] != "judge_calibration_control"]
    shifted = [spec for spec in specs if spec["split"] == "judge_calibration_control"]
    assert all(NOMINAL_SCALE[0] <= float(spec["kla"][2] / 240.0) <= NOMINAL_SCALE[1] for spec in nominal)
    assert all(float(spec["kla"][2] / 240.0) >= CONTROL_SCALE[0] for spec in shifted)


def test_start_states_differ_by_minute() -> None:
    pytest.importorskip("bsm2_python")
    from scs.systems.bsm1 import dry_start_states, load_dry_influent

    plant_state = _state_after_short_rollout()
    states = dry_start_states(load_dry_influent(), [0, 30], plant_state, warmup_minutes=0)
    assert set(states) == {0, 30}
    assert np.allclose(states[0]["reactor5"], plant_state["reactor5"])
    assert not np.allclose(states[30]["reactor5"], states[0]["reactor5"])


def test_pretest_dataset_round_trip(tmp_path) -> None:
    pytest.importorskip("bsm2_python")
    from scs.data.schemas import load_dataset, save_dataset
    from scs.systems.bsm1 import generate_bsm1_pretest_dataset

    from scs.systems.bsm1 import pretest_specs

    plant_state = _state_after_short_rollout()
    counts = {"n_model_train": 1, "n_calibration_id": 1, "n_calibration_control": 1, "horizon": 4, "history": 2}
    starts = [int(spec["start_minute"]) for spec in pretest_specs(seed=1, **counts)]
    dataset, manifest = generate_bsm1_pretest_dataset(
        seed=1,
        start_states={start: plant_state for start in starts},
        **counts,
    )
    assert manifest["reserved_minutes"] == [int(5.5 * 1440), 7 * 1440]
    assert manifest["unique_dry_days"] == 7
    assert [repair["raw"] for repair in manifest["rain_repairs"]] == ["30.044.50", "27.446.67", "26.880.33"]
    assert manifest["confirmatory_test_generated"] is False
    assert "judge_test_id" not in dataset
    assert manifest["weather"]["rain"] == "repaired"
    assert manifest["weather"]["storm"] == "missing"
    save_dataset(dataset, tmp_path)
    loaded = load_dataset(tmp_path)
    assert loaded["model_train"].measured_history.shape[0] == 1
    assert loaded["model_train"].measured_history.shape[-1] == 5
    assert np.isfinite(loaded["judge_calibration_control"].states).all()


def _state_after_short_rollout() -> dict:
    import numpy as np
    from bsm2_python.bsm1_ol import BSM1OL

    from scs.systems.bsm1 import NOMINAL_KLA, PUBLISHED_CONSTANT_INFLUENT, capture_plant

    times = np.arange(6) / 1440.0
    influent = np.repeat(PUBLISHED_CONSTANT_INFLUENT[None, :], len(times), axis=0)
    plant = BSM1OL(data_in=np.column_stack([times, influent]), timestep=1.0 / 1440.0)
    for index in range(3):
        plant.step(index, klas=NOMINAL_KLA)
    return capture_plant(plant)


def test_bsm1_effluent_limit_is_one_bit_per_quantity() -> None:
    import yaml

    from scs.experiments.v2 import trajectory_bad_event, trajectory_event_flags
    from scs.systems.bsm1 import EFFLUENT_OFFSET, STATE_DIM

    config = yaml.safe_load(open("configs/v3/bsm1_events.yaml", encoding="utf-8"))
    states = np.zeros((3, STATE_DIM))
    assert trajectory_bad_event("bsm1", states, config) is False
    states[1, EFFLUENT_OFFSET + 9] = 4.1
    flags = trajectory_event_flags("bsm1", states, config)
    assert flags["s_nh_above_limit"] is True
    assert flags["total_n_above_limit"] is False
    assert trajectory_bad_event("bsm1", states, config) is True


def test_effluent_measurements_match_the_library() -> None:
    pytest.importorskip("bsm2_python")
    from bsm2_python.bsm2.plantperformance import PlantPerformance
    import bsm2_python.bsm2.init.plantperformanceinit_bsm1 as pp_init

    from scs.systems.bsm1 import EFFLUENT_OFFSET, STATE_DIM, effluent_measurements

    effluent = np.array([30, 2, 5, 40, 10, 1, 1, 0.5, 8, 1.7, 0.7, 1, 7, 12, 18000, 15, 0, 0, 0, 0, 0], dtype=float)
    states = np.zeros(STATE_DIM)
    states[EFFLUENT_OFFSET : EFFLUENT_OFFSET + 21] = effluent
    performance = PlantPerformance(pp_init.PP_PAR)
    library = np.asarray(performance.advanced_quantities(effluent, ("totalN", "COD", "X_TSS", "BOD5e")), dtype=float).reshape(-1)
    measured = effluent_measurements(states)
    assert measured[0] == pytest.approx(effluent[9])
    assert measured[1:] == pytest.approx(library)


def test_bsm1_rollout_has_a_measured_history() -> None:
    pytest.importorskip("bsm2_python")
    from scs.systems.bsm1 import MEASURED_NAMES, STATE_DIM, influent_file_status, rollout_scenario

    status = influent_file_status()
    assert status["dry"] == "ok"
    assert status["rain"] == "repaired"
    assert status["storm"] == "missing"
    scenario = rollout_scenario(history=2, horizon=3)
    assert scenario.system_id == "bsm1"
    assert scenario.states.shape[2] == STATE_DIM
    assert scenario.measured_history is not None
    assert scenario.measured_history.shape == (1, 2, len(MEASURED_NAMES))
    assert np.isfinite(scenario.measured_history).all()


def test_dry_file_has_seven_unique_days() -> None:
    pytest.importorskip("bsm2_python")
    from scs.systems.bsm1 import DRY_MINUTES, load_dry_influent

    dry = load_dry_influent()
    assert dry.shape[0] * 15 == DRY_MINUTES
    # Every window must stay inside the 7 unique days. The packaged file repeats them,
    # so a window read from days 7 to 13 would be a copy of one from days 0 to 6.
    assert DRY_MINUTES == 7 * 1440


def test_rain_file_is_repaired_and_matches_the_dry_units() -> None:
    pytest.importorskip("bsm2_python")
    from scs.systems.bsm1 import RAIN_FLOW_COLUMN, load_dry_influent, load_rain_influent

    rain, repairs = load_rain_influent()
    assert [repair["row"] for repair in repairs] == [997, 999, 1000]
    flow = rain[:, RAIN_FLOW_COLUMN]
    # The repaired cells follow the straight line from 28013 to 26314 m3/d.
    assert flow[998:1002] == pytest.approx([28013.0, 27446.67, 26880.33, 26314.0], abs=0.01)
    assert float(np.median(flow)) > 1000.0
    # Days 0 to 7 of the rain file carry the dry flows, so the units agree.
    assert flow[:96].mean() == pytest.approx(load_dry_influent()[:96, RAIN_FLOW_COLUMN].mean(), rel=1e-3)
    # The one rain event is on days 8 to 10.
    daily = flow.reshape(14, 96).mean(axis=1)
    assert daily[8:11].min() > 1.3 * daily[:8].max()


def test_a_corrupted_rain_cell_outside_the_flow_column_is_refused(tmp_path, monkeypatch) -> None:
    pytest.importorskip("bsm2_python")
    from scs.systems import bsm1

    rows = ["0," + ",".join(["1.0"] * 21) for _ in range(5)]
    rows[2] = rows[2].replace("1.0", "1.0.5", 1)
    (tmp_path / "raininfluent.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    monkeypatch.setattr(bsm1, "_package_data_dir", lambda: tmp_path)
    with pytest.raises(RuntimeError, match="unexpected malformed"):
        bsm1.load_rain_influent()
