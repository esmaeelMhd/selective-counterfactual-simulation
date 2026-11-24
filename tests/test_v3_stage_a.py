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
    assert "| event_guard | two_tank | bad_event | 0.0500 | 1 | 0.1000 | 0.2000 | 0.3000 | 0.1000 |" in text
    assert "| 0.2000 |" in text
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


def test_descriptive_coverages_are_scored_but_stay_out_of_the_decision_tables(tmp_path: Path) -> None:
    from scs.experiments.v3 import interval_table

    path = _tiny_config(tmp_path / "tiny.yaml")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["descriptive_coverages"] = [0.5, 0.75]
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    run_v3_stage(path, tmp_path / "out")
    results = pd.read_csv(tmp_path / "out" / "stage_a_results.csv")
    roles = results.groupby("coverage")["coverage_role"].first().to_dict()
    assert roles == {0.05: "primary", 0.1: "primary", 0.5: "descriptive", 0.75: "descriptive"}
    # No judge or baseline was selected on a descriptive coverage: the simple baseline is the same one.
    assert results.groupby(["system_id", "seed", "model_id", "badness_target"])["calibration_simple_baseline"].nunique().max() == 1
    report = (tmp_path / "out" / "stage_a_report.md").read_text(encoding="utf-8")
    assert "## Descriptive coverages" in report and "part of any decision rule" in report


def test_the_guard_table_says_where_q3_has_no_room() -> None:
    from scs.experiments.v3 import _guard_lines

    def row(system, model, far, primary, simple):
        return {
            "system_id": system,
            "model_id": model,
            "judge_id": "event_guard",
            "eligible": True,
            "coverage": 0.05,
            "badness_target": "bad_event",
            "false_accept_rate": far,
            "primary_far": primary,
            "calibration_simple_far": simple,
        }

    text = _guard_lines(pd.DataFrame([row("full", "m", 0.01, 0.30, 0.20), row("floor", "m", 0.0, 0.01, 0.0)]))
    lines = {line.split("|")[2].strip(): line for line in text.splitlines() if line.startswith("| event_guard")}
    assert lines["full"].rstrip().endswith("| True |")
    assert lines["floor"].rstrip().endswith("| False |")
    assert "cannot be yes" in text


def test_the_interval_table_reads_primary_coverages_only() -> None:
    from scs.experiments.v3 import interval_table

    rows = _margin_rows("cstr", "bad_rmse", 0.4, [0.30, 0.31, 0.29])
    rows += [{**row, "coverage": 0.5} for row in _margin_rows("cstr", "bad_rmse", 0.4, [0.0, 0.1, 0.2])]
    for row in rows:
        row["coverage_role"] = "primary" if row["coverage"] == 0.05 else "descriptive"
    table = interval_table(pd.DataFrame(rows), {"bootstrap_seed": 1, "bootstrap_samples": 20, "interval_width_limit": 0.10})
    assert list(table["coverage"]) == [0.05]


def test_primary_ranker_selects_per_model_over_finite_signals_without_pooling_ineligible_models() -> None:
    from scs.experiments.v3 import eligibility_table, result_rows_for_frame

    rows = []
    # Two models on a BSM1-like system where invariant_residual and repair_amount are NaN.
    # On good_model: support_distance perfectly separates bad from good, while disagreement_score
    # is anti-informative noise. On hold_last: almost always bad with inverted support_distance and
    # zero uncertainty. Pooling across models or skipping candidate selection breaks good_model.
    for role in ("judge_calibration", "judge_test"):
        split = "judge_calibration_id" if role == "judge_calibration" else "judge_test_dry_holdout"
        for idx in range(20):
            good_bad = idx >= 10
            # On good_model, low support_distance (< 10) is always good; disagreement is reversed for
            # the lowest-support rows so an unselected rank-mean picks bad rows at 20% coverage.
            rows.append(
                {
                    "system_id": "bsm1",
                    "seed": 0,
                    "split": split,
                    "role": role,
                    "model_id": "good_model",
                    "rmse": 2.0 if good_bad else 0.1,
                    "event_mismatch": good_bad,
                    "support_distance": float(idx),
                    "uncertainty_score": np.nan,
                    "disagreement_score": float(20 - idx if idx < 10 else idx - 10),
                    "invariant_residual": np.nan,
                    "repair_amount": np.nan,
                }
            )
            # Poisoned hold_last rows with huge negative-correlated support_distance.
            rows.append(
                {
                    "system_id": "bsm1",
                    "seed": 0,
                    "split": split,
                    "role": role,
                    "model_id": "hold_last",
                    "rmse": 100.0 if idx < 18 else 0.05,
                    "event_mismatch": idx < 18,
                    "support_distance": float(100 - idx if idx < 18 else 500 + idx),
                    "uncertainty_score": np.nan,
                    "disagreement_score": float(idx),
                    "invariant_residual": np.nan,
                    "repair_amount": np.nan,
                }
            )
    frame = pd.DataFrame(rows)
    config = {
        "learned_models": ["good_model"],
        "eligibility_baselines": ["hold_last"],
        "primary_coverages": [0.20],
        "badness_targets": ["bad_rmse"],
        "rmse_threshold": 1.0,
        "event_guards": {"event_guard_no_invariant": ["risk_support_only", "risk_uncertainty_only", "risk_disagreement_only"]},
    }
    results = result_rows_for_frame(frame, config, eligibility_table(frame, config))
    good_primary = results[(results["model_id"] == "good_model") & (results["judge_id"] == "calibration_selected_candidate_ranker")]
    assert float(good_primary["false_accept_rate"].iloc[0]) == pytest.approx(0.0)


def test_deterministic_model_returns_nan_uncertainty_and_ensemble_reuses_member_rollouts() -> None:
    from scs.experiments.v3 import _rollout_and_uncertainty
    from scs.models.ensemble_mlp import EnsembleMLPModel
    from scs.models.linear_narx import LinearNARXModel

    rng = np.random.default_rng(0)
    batch = TrajectoryBatch(
        states=rng.normal(size=(8, 5, 2)),
        actions=rng.normal(size=(8, 4, 1)),
        disturbances=rng.normal(size=(8, 4, 2)),
        scenario_type=["id"] * 8,
        split="model_train",
        system_id="two_tank",
    )
    linear = LinearNARXModel(random_state=0)
    linear.fit(batch)
    ensemble = EnsembleMLPModel(random_state=0, n_members=5, max_iter=10)
    ensemble.fit(batch)

    calls = [0] * len(ensemble.members)
    for idx, member in enumerate(ensemble.members):
        original = member.predict_rollouts

        def wrapped(*args, _orig=original, _idx=idx, **kwargs):
            calls[_idx] += 1
            return _orig(*args, **kwargs)

        member.predict_rollouts = wrapped  # type: ignore[method-assign]

    lin_pred, lin_unc = _rollout_and_uncertainty(
        linear, batch.states[:, 0], batch.actions, batch.disturbances, n_samples=5
    )
    ens_pred, ens_unc = _rollout_and_uncertainty(
        ensemble, batch.states[:, 0], batch.actions, batch.disturbances, n_samples=5
    )
    assert lin_pred.shape == batch.states.shape
    assert np.isnan(lin_unc).all()
    assert ens_pred.shape == batch.states.shape
    assert np.isfinite(ens_unc).all() and (ens_unc >= 0.0).all()
    # Every ensemble member runs exactly once (prediction mean and uncertainty share the rollouts).
    assert calls == [1, 1, 1, 1, 1]


def test_disagreement_cohort_excludes_hold_last_and_ineligible_models() -> None:
    from scs.experiments.v3 import disagreement_cohort

    eligibility = pd.DataFrame(
        [
            {"system_id": "bsm1", "model_id": "hold_last", "eligible": False},
            {"system_id": "bsm1", "model_id": "linear_narx", "eligible": False},
            {"system_id": "bsm1", "model_id": "mlp_state_space", "eligible": False},
            {"system_id": "bsm1", "model_id": "lstm_world", "eligible": True},
            {"system_id": "bsm1", "model_id": "ensemble_lstm", "eligible": True},
        ]
    )
    config = {
        "models": ["hold_last", "linear_narx", "mlp_state_space", "lstm_world", "ensemble_lstm"],
        "learned_models": ["mlp_state_space", "lstm_world", "ensemble_lstm"],
    }
    assert disagreement_cohort("bsm1", eligibility, config) == ["lstm_world", "ensemble_lstm"]
    # When fewer than 2 learned models are eligible, fall back to learned models (never hold_last).
    none_eligible = eligibility.assign(eligible=False)
    cohort = disagreement_cohort("bsm1", none_eligible, config)
    assert "hold_last" not in cohort and set(cohort) == {"mlp_state_space", "lstm_world", "ensemble_lstm"}


def test_standardized_state_rmse_balances_channel_scales_and_skips_constant_channels() -> None:
    from scs.experiments.v3 import fit_state_scale, scaled_scenario_rmse

    rng = np.random.default_rng(7)
    train_states = np.zeros((10, 6, 3), dtype=float)
    train_states[:, :, 0] = 15.0  # constant channel (like BSM1 temperature/dummies)
    train_states[:, :, 1] = rng.normal(0.0, 1.0, size=(10, 6))
    train_states[:, :, 2] = rng.normal(0.0, 1000.0, size=(10, 6))
    active, std = fit_state_scale(train_states)
    assert list(active) == [False, True, True]
    true_states = np.zeros((2, 6, 3), dtype=float)
    pred_states = true_states.copy()
    pred_states[:, :, 0] = 999.0  # constant channel ignored
    pred_states[:, :, 1] = std[0]  # +1 std error
    pred_states[:, :, 2] = std[1]  # +1 std error
    scores = scaled_scenario_rmse(pred_states, true_states, (active, std))
    assert scores == pytest.approx([1.0, 1.0])


def test_stage_b_report_includes_both_coverages_relative_margins_q3_cis_and_freeze_metadata(tmp_path: Path) -> None:
    from scs.experiments.v3 import _guard_lines, _write_report, interval_table

    rows = []
    for coverage in (0.05, 0.10):
        for seed in range(5):
            for judge_id, far in (
                ("random_baseline", 0.40),
                ("calibration_selected_candidate_ranker", 0.20 - 0.01 * seed),
                ("event_guard_no_invariant", 0.05 + 0.005 * seed),
            ):
                abs_margin = 0.40 - far
                rows.append(
                    {
                        "system_id": "bsm1",
                        "seed": seed,
                        "model_id": "ensemble_lstm",
                        "eligible": True,
                        "badness_target": "bad_event",
                        "coverage": coverage,
                        "coverage_role": "primary",
                        "judge_id": judge_id,
                        "false_accept_rate": far,
                        "random_far": 0.40,
                        "absolute_margin": abs_margin,
                        "relative_margin": abs_margin / 0.40,
                        "primary_far": 0.20 - 0.01 * seed,
                        "calibration_simple_far": 0.25 - 0.01 * seed,
                        "bad_share": 0.40,
                        "accepted_count": int(coverage * 2000),
                        "n_scenarios": 2000,
                        "calibration_simple_baseline": "support_only",
                        "rmse_threshold": 1.5,
                    }
                )
    results = pd.DataFrame(rows)
    config = {
        "stage": "B",
        "label": "confirmatory",
        "systems": ["bsm1"],
        "learned_models": ["ensemble_lstm"],
        "primary_coverages": [0.05, 0.10],
        "rmse_threshold": {"bsm1": 1.5},
        "rmse_metric": "standardized_state",
        "seeds": [0, 1, 2, 3, 4],
        "min_test_scenarios": 2000,
        "bootstrap_samples": 100,
        "bootstrap_seed": 2025,
        "interval_width_limit": 0.10,
        "data": {"horizon": 480},
        "model_runtime": {"ensemble_members": 5, "lstm_epochs": 120},
    }
    eligibility = pd.DataFrame(
        [
            {
                "system_id": "bsm1",
                "model_id": "ensemble_lstm",
                "id_rmse": 0.12,
                "baseline_hold_last_id_rmse": 0.86,
                "baseline_linear_narx_id_rmse": 0.21,
                "eligible": True,
            }
        ]
    )
    intervals = interval_table(results, config)
    assert sorted(intervals["coverage"].tolist()) == [0.05, 0.10]
    assert "mean_relative_margin" in intervals.columns
    assert intervals["mean_relative_margin"].iloc[0] == pytest.approx(0.55, abs=0.01)

    guard_text = _guard_lines(results, config=config)
    assert "0.0500" in guard_text and "0.1000" in guard_text
    assert "ci_low_vs_primary" in guard_text and "ci_low_vs_simple" in guard_text
    assert "rel_vs_primary" in guard_text and "rel_vs_simple" in guard_text

    report_path = tmp_path / "stage_b_report.md"
    metadata = {"git_commit": "abc1234", "protocol_sha256": "deadbeef" * 8}
    _write_report(report_path, eligibility, intervals, results, config, metadata=metadata)
    report = report_path.read_text(encoding="utf-8")
    assert "0.0500" in report and "0.1000" in report
    assert "abc1234" in report and ("deadbeef" * 8) in report
    assert "Exploratory only." not in report
    assert "A freeze is a later commit." not in report
    assert "Stage A check" not in report


def test_primary_ranker_breaks_zero_far_ties_using_calibration_discrimination_not_raw_mean() -> None:
    from scs.experiments.v3 import _fit_calibrated_ranker

    # On calibration rows, both support_distance and repair_amount achieve 0.0 FAR at 10% coverage,
    # because the bottom 10% of both are good. However, repair_amount is mostly zero (tiny raw mean)
    # and fails above 20% coverage, whereas support_distance cleanly ranks all bad rows highest.
    n = 40
    idx = np.arange(n, dtype=float)
    bad = idx >= 20
    cal = pd.DataFrame(
        {
            "bad_label": bad,
            "support_distance": 10.0 + idx,
            "uncertainty_score": np.nan,
            "disagreement_score": np.nan,
            "invariant_residual": np.nan,
            "repair_amount": np.where(idx < 38, 0.0, 1.0),
        }
    )
    # Same distribution on test, evaluated at 50% coverage where support_distance gets 0.0 FAR
    # and nearly-constant repair_amount gets poor FAR (0.5 on tied zeros / order-dependent).
    test = cal.iloc[::-1].reset_index(drop=True)
    scores = _fit_calibrated_ranker(cal, test, primary_coverages=[0.10])
    accepted = np.argsort(scores, kind="mergesort")[:20]
    far_at_50 = float(np.mean(test["bad_label"].to_numpy(dtype=bool)[accepted]))
    assert far_at_50 == pytest.approx(0.0)


def test_descriptive_bad_share_weights_each_eligible_model_equally() -> None:
    from scs.experiments.v3 import _descriptive_lines

    # Model A has 3 judges (bad_share=0.10); Model B has 1 judge (bad_share=0.90).
    # Equal model weighting must give bad_share = 0.5000, not (3*0.10 + 1*0.90)/4 = 0.3000.
    rows = [
        {"system_id": "cstr", "badness_target": "bad_rmse", "coverage": 0.25, "coverage_role": "descriptive", "eligible": True, "model_id": "A", "judge_id": "calibration_selected_candidate_ranker", "false_accept_rate": 0.1, "calibration_simple_far": 0.1, "bad_share": 0.1},
        {"system_id": "cstr", "badness_target": "bad_rmse", "coverage": 0.25, "coverage_role": "descriptive", "eligible": True, "model_id": "A", "judge_id": "uncertainty_only", "false_accept_rate": 0.1, "calibration_simple_far": 0.1, "bad_share": 0.1},
        {"system_id": "cstr", "badness_target": "bad_rmse", "coverage": 0.25, "coverage_role": "descriptive", "eligible": True, "model_id": "A", "judge_id": "event_guard_no_invariant", "false_accept_rate": 0.1, "calibration_simple_far": 0.1, "bad_share": 0.1},
        {"system_id": "cstr", "badness_target": "bad_rmse", "coverage": 0.25, "coverage_role": "descriptive", "eligible": True, "model_id": "B", "judge_id": "calibration_selected_candidate_ranker", "false_accept_rate": 0.2, "calibration_simple_far": 0.2, "bad_share": 0.9},
    ]
    text = _descriptive_lines(pd.DataFrame(rows))
    assert "| 0.5000 |" in text


def test_test_risk_curve_is_tie_aware_and_permutation_invariant() -> None:
    from scs.experiments.v3 import eligibility_table, result_rows_for_frame

    # Split-ordered rows: 10 good ID rows followed by 10 bad OOD rows (bad_share = 0.5).
    # repair_amount is identically 0.0 everywhere, and support_distance is tied across blocks.
    def make_frame(reverse_test: bool) -> pd.DataFrame:
        rows = []
        for role in ("judge_calibration", "judge_test"):
            split = "judge_calibration_id" if role == "judge_calibration" else "judge_test_id"
            order = list(range(20))[::-1] if (role == "judge_test" and reverse_test) else list(range(20))
            for idx in order:
                bad = idx >= 10
                rows.append(
                    {
                        "system_id": "two_tank",
                        "seed": 0,
                        "split": split,
                        "role": role,
                        "model_id": "good_model",
                        "rmse": 2.0 if bad else 0.01,
                        "event_mismatch": bad,
                        "support_distance": 1.0 if idx < 15 else 2.0,
                        "uncertainty_score": np.nan,
                        "disagreement_score": 0.1 if idx < 12 else 0.9,
                        "invariant_residual": 0.2 if idx < 14 else 0.8,
                        "repair_amount": 0.0,
                    }
                )
        return pd.DataFrame(rows)

    config = {
        "learned_models": ["good_model"],
        "eligibility_baselines": ["good_model"],
        "primary_coverages": [0.10, 0.25],
        "badness_targets": ["bad_rmse"],
        "rmse_threshold": 0.5,
        "event_guards": {"event_guard": ["risk_invariant_only", "risk_disagreement_only", "risk_support_only"]},
    }
    fwd = make_frame(reverse_test=False)
    rev = make_frame(reverse_test=True)
    res_fwd = result_rows_for_frame(fwd, config, eligibility_table(fwd, config))
    res_rev = result_rows_for_frame(rev, config, eligibility_table(rev, config))

    # Constant all-zero repair_only must get expected FAR == bad_share (0.5), never 0.0 from row order.
    rep_fwd = res_fwd[(res_fwd["judge_id"] == "repair_only") & np.isclose(res_fwd["coverage"], 0.10)]
    assert float(rep_fwd["false_accept_rate"].iloc[0]) == pytest.approx(0.5)

    # Permuting test rows must leave every non-random judge FAR identical.
    for judge_id in ("repair_only", "support_only", "disagreement_only", "invariant_only", "event_guard", "calibration_selected_candidate_ranker"):
        f_vals = res_fwd[res_fwd["judge_id"] == judge_id].sort_values("coverage")["false_accept_rate"].to_numpy()
        r_vals = res_rev[res_rev["judge_id"] == judge_id].sort_values("coverage")["false_accept_rate"].to_numpy()
        assert f_vals == pytest.approx(r_vals)


def test_calibration_selection_does_not_peek_at_test_and_nan_rmse_is_bad() -> None:
    from scs.experiments.v3 import eligibility_table, result_rows_for_frame

    rows = []
    for role in ("judge_calibration", "judge_test"):
        split = "judge_calibration_id" if role == "judge_calibration" else "judge_test_id"
        for idx in range(20):
            bad = idx >= 10
            # On one test row, rmse and support_distance diverge to NaN.
            diverged = role == "judge_test" and idx == 19
            rows.append(
                {
                    "system_id": "bsm1",
                    "seed": 0,
                    "split": split,
                    "role": role,
                    "model_id": "good_model",
                    "rmse": np.nan if diverged else (2.0 if bad else 0.1),
                    "event_mismatch": bad,
                    "support_distance": np.nan if diverged else float(idx),
                    "uncertainty_score": np.nan,
                    "disagreement_score": float(20 - idx),
                    "invariant_residual": np.nan,
                    "repair_amount": np.nan,
                }
            )
    frame = pd.DataFrame(rows)
    config = {
        "learned_models": ["good_model"],
        "eligibility_baselines": ["good_model"],
        "primary_coverages": [0.20],
        "badness_targets": ["bad_rmse"],
        "rmse_threshold": 1.0,
        "event_guards": {"event_guard_no_invariant": ["risk_support_only", "risk_uncertainty_only", "risk_disagreement_only"]},
    }
    results = result_rows_for_frame(frame, config, eligibility_table(frame, config))
    # All 10 rows idx=10..19 (including the NaN rmse row) are bad -> bad_share == 0.50.
    assert float(results["bad_share"].iloc[0]) == pytest.approx(0.50)
    # support_only must NOT be dropped just because one test row has NaN support_distance;
    # the NaN test row is ranked last (+inf) so low-coverage acceptance still gets 0.0 FAR.
    sup = results[results["judge_id"] == "support_only"]
    assert not sup.empty and float(sup["false_accept_rate"].iloc[0]) == pytest.approx(0.0)
    primary = results[results["judge_id"] == "calibration_selected_candidate_ranker"]
    assert float(primary["false_accept_rate"].iloc[0]) == pytest.approx(0.0)


def test_interval_table_uses_ratio_of_means_for_relative_margin() -> None:
    from scs.experiments.v3 import interval_table

    # Seed 0: random_far=0.10, margin=0.05 (ratio 0.50); Seed 1: random_far=0.30, margin=0.30 (ratio 1.00).
    # Ratio of means = (0.05 + 0.30) / (0.10 + 0.30) = 0.875 (NOT mean of ratios 0.75).
    rows = [
        {"system_id": "cstr", "badness_target": "bad_rmse", "coverage": 0.05, "coverage_role": "primary", "seed": 0, "model_id": "m", "judge_id": "calibration_selected_candidate_ranker", "eligible": True, "absolute_margin": 0.05, "random_far": 0.10, "relative_margin": 0.50, "bad_share": 0.10},
        {"system_id": "cstr", "badness_target": "bad_rmse", "coverage": 0.05, "coverage_role": "primary", "seed": 1, "model_id": "m", "judge_id": "calibration_selected_candidate_ranker", "eligible": True, "absolute_margin": 0.30, "random_far": 0.30, "relative_margin": 1.00, "bad_share": 0.30},
    ]
    table = interval_table(pd.DataFrame(rows), {"bootstrap_seed": 1, "bootstrap_samples": 20, "interval_width_limit": 0.10})
    assert float(table["mean_relative_margin"].iloc[0]) == pytest.approx(0.875)


def test_rollout_supports_three_arg_custom_model_without_predict_rollouts() -> None:
    from scs.experiments.v3 import _rollout_and_uncertainty
    from scs.models.user_model import UserSimulatorModel

    class ThreeArgUserModel(UserSimulatorModel):
        model_id = "custom_3arg"

        def fit(self, train: TrajectoryBatch) -> None:
            return None

        def predict_rollout(self, initial_state: np.ndarray, actions: np.ndarray, disturbances: np.ndarray) -> np.ndarray:
            states = np.empty((len(actions) + 1, initial_state.shape[0]), dtype=float)
            states[0] = initial_state
            for step in range(len(actions)):
                states[step + 1] = states[step] + 0.5
            return states

    model = ThreeArgUserModel()
    initial = np.zeros((3, 2))
    actions = np.zeros((3, 4, 1))
    disturbances = np.zeros((3, 4, 1))
    pred, unc = _rollout_and_uncertainty(model, initial, actions, disturbances, n_samples=5)
    assert pred.shape == (3, 5, 2)
    assert pred[:, -1, :] == pytest.approx(2.0)
    assert np.isnan(unc).all()


def test_guard_lines_reports_configured_guard_undefined_on_all_systems_and_dynamic_baselines(tmp_path: Path) -> None:
    from scs.experiments.v3 import _guard_lines, _write_report

    results = pd.DataFrame(
        [
            {
                "system_id": "bsm1",
                "seed": 0,
                "model_id": "m",
                "eligible": True,
                "badness_target": "bad_event",
                "coverage": 0.05,
                "coverage_role": "primary",
                "judge_id": "calibration_selected_candidate_ranker",
                "false_accept_rate": 0.1,
                "random_far": 0.4,
                "absolute_margin": 0.3,
                "relative_margin": 0.75,
                "primary_far": 0.1,
                "calibration_simple_far": 0.2,
                "bad_share": 0.4,
            }
        ]
    )
    config = {
        "stage": "A",
        "label": "exploratory",
        "systems": ["bsm1"],
        "learned_models": ["m"],
        "eligibility_baselines": ["hold_last"],
        "primary_coverages": [0.05],
        "rmse_threshold": {"bsm1": 0.5},
        "seeds": [0],
        "min_test_scenarios": 1,
        "bootstrap_samples": 10,
        "bootstrap_seed": 1,
        "interval_width_limit": 0.10,
        "data": {"horizon": 4},
        "model_runtime": {"ensemble_members": 5},
        "event_guards": {"custom_full_state_guard": ["risk_invariant_only"]},
    }
    text = _guard_lines(results, config=config)
    assert "`custom_full_state_guard` is not defined on: bsm1" in text

    eligibility = pd.DataFrame(
        [{"system_id": "bsm1", "model_id": "m", "id_rmse": 0.1, "baseline_hold_last_id_rmse": 0.9, "eligible": True}]
    )
    report_path = tmp_path / "custom_baseline_report.md"
    _write_report(report_path, eligibility, pd.DataFrame(), results, config)
    assert "baseline_hold_last_id_rmse" in report_path.read_text(encoding="utf-8")
