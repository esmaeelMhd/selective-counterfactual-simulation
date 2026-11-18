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
