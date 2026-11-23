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
            {"system_id": "two_tank", "split": "judge_calibration_id", "model_id": "hold_last", "rmse": 1.0},
            {"system_id": "two_tank", "split": "judge_calibration_id", "model_id": "linear_narx", "rmse": 0.2},
            {"system_id": "two_tank", "split": "judge_calibration_id", "model_id": "mlp_state_space", "rmse": 0.3},
            {"system_id": "two_tank", "split": "judge_calibration_id", "model_id": "ensemble_mlp", "rmse": 0.1},
            {"system_id": "two_tank", "split": "judge_calibration_ood", "model_id": "ensemble_mlp", "rmse": 5.0},
            # Test rows must not move eligibility, however good they look.
            {"system_id": "two_tank", "split": "judge_test_id", "model_id": "mlp_state_space", "rmse": 1e-9},
        ]
    )
    config = {
        "learned_models": ["mlp_state_space", "ensemble_mlp"],
        "eligibility_baselines": ["hold_last", "linear_narx"],
    }
    table = eligibility_table(frame, config)
    eligible = set(table.loc[table["eligible"], "model_id"])
    assert eligible == {"ensemble_mlp"}


def test_eligibility_refuses_a_judge_test_split() -> None:
    frame = pd.DataFrame(
        [{"system_id": "two_tank", "split": "judge_test_id", "model_id": "hold_last", "rmse": 1.0}]
    )
    config = {
        "learned_models": [],
        "eligibility_baselines": ["hold_last"],
        "eligibility_split": "judge_test_id",
    }
    with pytest.raises(ValueError, match="judge-test"):
        eligibility_table(frame, config)


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


def _margin_rows(system: str, target: str, bad_share: float, margins: list[float]) -> list[dict]:
    return [
        {
            "system_id": system,
            "badness_target": target,
            "coverage": 0.05,
            "seed": seed,
            "judge_id": "calibration_selected_candidate_ranker",
            "eligible": True,
            "absolute_margin": margin,
            "bad_share": bad_share,
        }
        for seed, margin in enumerate(margins)
    ]


def test_rmse_threshold_can_differ_by_system() -> None:
    from scs.experiments.v3 import rmse_threshold_for

    config = {"rmse_threshold": {"two_tank": 0.02, "cstr": 0.3}}
    assert rmse_threshold_for(config, "two_tank") == 0.02
    assert rmse_threshold_for(config, "cstr") == 0.3
    assert rmse_threshold_for({"rmse_threshold": 0.05}, "cstr") == 0.05


def test_config_requires_a_threshold_for_every_system(tmp_path: Path) -> None:
    path = _tiny_config(tmp_path / "tiny.yaml")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["rmse_threshold"] = {"cstr": 0.1}
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="two_tank"):
        load_v3_config(path)


def test_a_target_with_no_bad_scenarios_is_degenerate_and_not_a_pass() -> None:
    from scs.experiments.v3 import interval_table, precision_check

    config = {"bootstrap_seed": 1, "bootstrap_samples": 50, "interval_width_limit": 0.10}
    rows = _margin_rows("two_tank", "bad_event", 0.0, [0.0, 0.0, 0.0]) + _margin_rows(
        "two_tank", "bad_rmse", 0.4, [0.30, 0.31, 0.29]
    )
    table = interval_table(pd.DataFrame(rows), config)
    flags = dict(zip(table["badness_target"], table["degenerate"]))
    assert flags == {"bad_event": True, "bad_rmse": False}
    passed, degenerate = precision_check(table)
    assert passed and degenerate == 1


def test_precision_needs_an_informative_row_on_every_system() -> None:
    from scs.experiments.v3 import interval_table, precision_check

    config = {"bootstrap_seed": 1, "bootstrap_samples": 50, "interval_width_limit": 0.10}
    rows = _margin_rows("two_tank", "bad_rmse", 0.4, [0.30, 0.31, 0.29]) + _margin_rows(
        "cstr", "bad_event", 1.0, [0.0, 0.0, 0.0]
    )
    passed, degenerate = precision_check(interval_table(pd.DataFrame(rows), config))
    assert not passed
    assert degenerate == 1


def test_a_wide_informative_interval_fails_the_check() -> None:
    from scs.experiments.v3 import interval_table, precision_check

    config = {"bootstrap_seed": 1, "bootstrap_samples": 200, "interval_width_limit": 0.10}
    rows = _margin_rows("two_tank", "bad_rmse", 0.4, [0.0, 0.9, 0.1, 0.8, 0.2, 0.7])
    passed, _degenerate = precision_check(interval_table(pd.DataFrame(rows), config))
    assert not passed


def test_guard_table_counts_only_eligible_models_and_names_missing_guards() -> None:
    from scs.experiments.v3 import _guard_lines

    def row(system, model, judge, eligible, far, primary, simple):
        return {
            "system_id": system,
            "model_id": model,
            "judge_id": judge,
            "eligible": eligible,
            "coverage": 0.05,
            "badness_target": "bad_event",
            "false_accept_rate": far,
            "primary_far": primary,
            "calibration_simple_far": simple,
        }

    results = pd.DataFrame(
        [
            row("two_tank", "good", "event_guard", True, 0.10, 0.20, 0.30),
            row("two_tank", "hold_last", "event_guard", False, 1.00, 1.00, 1.00),
            row("bsm1", "good", "calibration_selected_candidate_ranker", True, 0.2, 0.2, 0.2),
            row("cstr", "weak", "event_guard", False, 0.5, 0.5, 0.5),
        ]
    )
    text = _guard_lines(results)
    assert "| event_guard | two_tank | bad_event | 1 | 0.1000 | 0.2000 | 0.3000 | 0.1000 | 0.2000 |" in text
    assert "`event_guard` is not defined on: bsm1" in text
    assert "`event_guard` has no eligible model, so no row, on: cstr" in text


def test_guards_come_from_the_config_with_a_default() -> None:
    from scs.experiments.v3 import event_guard_definitions

    assert event_guard_definitions({"event_guard_inputs": ["risk_support_only"]}) == {"event_guard": ["risk_support_only"]}
    named = {"event_guards": {"a": ["risk_support_only"], "b": ["risk_uncertainty_only", "risk_support_only"]}}
    assert event_guard_definitions(named) == {"a": ["risk_support_only"], "b": ["risk_uncertainty_only", "risk_support_only"]}


def test_the_guard_without_invariant_runs_where_the_v2_guard_cannot() -> None:
    rows = []
    for role, bad in (("judge_calibration", True), ("judge_calibration", False), ("judge_test", True), ("judge_test", False)):
        for index in range(4):
            rows.append(
                {
                    "system_id": "bsm1",
                    "seed": 0,
                    "split": "judge_test_dry_holdout" if role == "judge_test" else "judge_calibration_id",
                    "role": role,
                    "model_id": "hold_last",
                    "rmse": 0.2 if bad else 0.01,
                    "event_mismatch": bad,
                    "support_distance": 1.0 + index if bad else 0.1 + index / 10,
                    "uncertainty_score": 0.5 + index / 10 if bad else 0.2,
                    "disagreement_score": 0.4 if bad else 0.1 + index / 20,
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
        "event_guards": {
            "event_guard": ["risk_invariant_only", "risk_disagreement_only", "risk_support_only"],
            "event_guard_no_invariant": ["risk_support_only", "risk_uncertainty_only", "risk_disagreement_only"],
        },
    }
    from scs.experiments.v3 import eligibility_table, result_rows_for_frame

    results = result_rows_for_frame(frame, config, eligibility_table(frame, config))
    judges = set(results["judge_id"])
    assert "event_guard_no_invariant" in judges
    assert "event_guard" not in judges


def test_thin_models_are_counted_inside_a_row_that_is_not_degenerate() -> None:
    from scs.experiments.v3 import interval_table

    rows = []
    for model, share in (("strong", 0.0), ("weak", 0.6)):
        for seed, margin in enumerate([0.1, 0.12, 0.11]):
            rows.append(
                {
                    "system_id": "bsm1",
                    "badness_target": "bad_rmse",
                    "coverage": 0.05,
                    "seed": seed,
                    "model_id": model,
                    "judge_id": "calibration_selected_candidate_ranker",
                    "eligible": True,
                    "absolute_margin": 0.0 if model == "strong" else margin,
                    "bad_share": share,
                }
            )
    table = interval_table(pd.DataFrame(rows), {"bootstrap_seed": 1, "bootstrap_samples": 20, "interval_width_limit": 0.10})
    row = table.iloc[0]
    assert bool(row["degenerate"]) is False
    assert (int(row["n_models"]), int(row["thin_models"])) == (2, 1)
