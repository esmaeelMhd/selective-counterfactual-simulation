from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from scs.experiments.v3 import (
    _relative_margin,
    bootstrap_interval,
    eligibility_table,
    load_v3_config,
    run_v3_stage,
)
from scs.models.linear_narx import LinearNARXModel
from scs.systems.base import TrajectoryBatch


def _tiny_config(path: Path) -> Path:
    config = {
        "stage": "A",
        "label": "exploratory",
        "event_config": "configs/v2/v2_event_targets.yaml",
        "systems": ["two_tank"],
        "models": ["hold_last", "linear_narx", "mlp_state_space", "ensemble_mlp"],
        "learned_models": ["mlp_state_space", "ensemble_mlp"],
        "eligibility_baselines": ["hold_last", "linear_narx"],
        "primary_coverages": [0.05, 0.10],
        "badness_targets": ["bad_rmse", "bad_event"],
        "rmse_threshold": 0.05,
        "seeds": [0],
        "event_guard_inputs": ["risk_invariant_only", "risk_disagreement_only", "risk_support_only"],
        "bootstrap_samples": 20,
        "bootstrap_seed": 2025,
        "interval_width_limit": 0.10,
        "min_test_scenarios": 1,
        "data": {
            "horizon": 4,
            "dt": 0.1,
            "n_model_train": 4,
            "n_calibration_id": 2,
            "n_calibration_ood": 2,
            "n_test_id": 3,
            "n_test_ood": 2,
            "uncertainty_samples": 1,
        },
        "model_runtime": {
            "mlp_max_iter": 8,
            "ensemble_members": 5,
            "ensemble_mlp_max_iter": 6,
            "gradient_boosting_max_iter": 8,
        },
    }
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_stage_a_rejects_a_small_ensemble(tmp_path: Path) -> None:
    path = _tiny_config(tmp_path / "tiny.yaml")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["model_runtime"]["ensemble_members"] = 2
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="at least 5"):
        load_v3_config(path)


def test_eligibility_requires_both_baselines() -> None:
    frame = pd.DataFrame(
        [
            {"system_id": "two_tank", "split": "judge_test_id", "model_id": "hold_last", "rmse": 1.0},
            {"system_id": "two_tank", "split": "judge_test_id", "model_id": "linear_narx", "rmse": 0.2},
            {"system_id": "two_tank", "split": "judge_test_id", "model_id": "mlp_state_space", "rmse": 0.3},
            {"system_id": "two_tank", "split": "judge_test_id", "model_id": "ensemble_mlp", "rmse": 0.1},
            {"system_id": "two_tank", "split": "judge_test_ood", "model_id": "ensemble_mlp", "rmse": 5.0},
        ]
    )
    config = {
        "learned_models": ["mlp_state_space", "ensemble_mlp"],
        "eligibility_baselines": ["hold_last", "linear_narx"],
    }
    table = eligibility_table(frame, config)
    eligible = set(table.loc[table["eligible"], "model_id"])
    assert eligible == {"ensemble_mlp"}


def test_relative_margin_is_undefined_when_random_far_is_zero() -> None:
    assert math.isnan(_relative_margin(0.1, 0.0))
    assert _relative_margin(0.1, 0.2) == pytest.approx(0.5)


def test_one_seed_does_not_produce_an_interval() -> None:
    low, high = bootstrap_interval(np.array([0.2]), np.random.default_rng(0), 20)
    assert math.isnan(low)
    assert math.isnan(high)


def test_batch_rollout_matches_the_single_scenario_rollout() -> None:
    rng = np.random.default_rng(0)
    batch = TrajectoryBatch(
        states=rng.normal(size=(3, 5, 2)),
        actions=rng.normal(size=(3, 4, 1)),
        disturbances=rng.normal(size=(3, 4, 2)),
        scenario_type=["id", "id", "id"],
        split="model_train",
        system_id="two_tank",
    )
    model = LinearNARXModel(random_state=0)
    model.fit(batch)
    batched = model.predict_rollouts(batch.states[:, 0], batch.actions, batch.disturbances)
    for index in range(3):
        single = model.predict_rollout(batch.states[index, 0], batch.actions[index], batch.disturbances[index])
        assert batched[index] == pytest.approx(single)


def test_stage_a_margin_is_against_random(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path / "tiny.yaml")
    output = tmp_path / "stage_a"
    result = run_v3_stage(config, output)
    table = pd.read_csv(output / "stage_a_results.csv")
    report = (output / "stage_a_report.md").read_text(encoding="utf-8")
    assert result["rows"] == len(table)
    assert "baseline_judge" not in table.columns
    assert np.allclose(table["absolute_margin"], table["random_far"] - table["false_accept_rate"])
    assert "does not answer Q1, Q2, or Q3" in report
    assert "exploratory" in report
    random_rows = table[table["judge_id"] == "random_baseline"]
    assert np.allclose(random_rows["absolute_margin"], 0.0)
