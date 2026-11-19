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


def test_missing_signal_still_scores() -> None:
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
    assert "event_guard" in judges
    assert np.isfinite(results["false_accept_rate"]).all()


def test_rain_storm_and_confirmatory_test_are_refused() -> None:
    from scs.systems.bsm1 import assert_split_allowed

    for split in ("judge_test_id", "judge_test_rain", "judge_test_storm", "judge_test_control", "rain", "storm"):
        with pytest.raises(RuntimeError):
            assert_split_allowed(split)


def test_pretest_windows_do_not_overlap() -> None:
    from scs.systems.bsm1 import CONTROL_SCALE, NOMINAL_SCALE, pretest_specs

    horizon, history = 8, 2
    specs = pretest_specs(n_model_train=2, n_calibration_id=2, n_calibration_control=2, horizon=horizon, history=history, seed=0)
    starts = sorted(int(spec["start_minute"]) for spec in specs)
    assert len(starts) == len(set(starts))
    assert all(right - left >= horizon + history for left, right in zip(starts, starts[1:]))
    assert {spec["split"] for spec in specs} == {"model_train", "judge_calibration_id", "judge_calibration_control"}
    nominal = [spec for spec in specs if spec["split"] != "judge_calibration_control"]
    shifted = [spec for spec in specs if spec["split"] == "judge_calibration_control"]
    assert all(NOMINAL_SCALE[0] <= float(spec["kla"][2] / 240.0) <= NOMINAL_SCALE[1] for spec in nominal)
    assert all(float(spec["kla"][2] / 240.0) >= CONTROL_SCALE[0] for spec in shifted)


def test_pretest_dataset_round_trip(tmp_path) -> None:
    pytest.importorskip("bsm2_python")
    from scs.data.schemas import load_dataset, save_dataset
    from scs.systems.bsm1 import generate_bsm1_pretest_dataset

    plant_state = _state_after_short_rollout()
    dataset, manifest = generate_bsm1_pretest_dataset(
        n_model_train=1,
        n_calibration_id=1,
        n_calibration_control=1,
        horizon=4,
        history=2,
        seed=1,
        steady_state=plant_state,
    )
    assert manifest["confirmatory_test_generated"] is False
    assert "judge_test_id" not in dataset
    assert manifest["weather"]["rain"] != "ok"
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
    assert status["rain"] != "ok"
    assert status["storm"] == "missing"
    scenario = rollout_scenario(history=2, horizon=3)
    assert scenario.system_id == "bsm1"
    assert scenario.states.shape[2] == STATE_DIM
    assert scenario.measured_history is not None
    assert scenario.measured_history.shape == (1, 2, len(MEASURED_NAMES))
    assert np.isfinite(scenario.measured_history).all()
