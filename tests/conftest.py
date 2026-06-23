from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_SEARCH_PATH = getattr(sys, "path")
if str(REPO_ROOT) not in PYTHON_SEARCH_PATH:
    PYTHON_SEARCH_PATH[:0] = [str(REPO_ROOT)]

from scs.reports.failure_analysis import PLOT_JUDGES, build_failure_table


def write_failure_results_fixture(path: Path, n_per_split: int = 6, threshold: float = 0.15) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    splits = ["id_test", "ood_action_magnitude", "ood_inflow_spike", "ood_combined"]
    scenario_types = {
        "id_test": "normal_policy",
        "ood_action_magnitude": "held_out_action_magnitude",
        "ood_inflow_spike": "inflow_spike",
        "ood_combined": "combined_intervention",
    }
    rows = []
    rng = np.random.default_rng(123)
    for model_idx, model_id in enumerate(["hold_last", "linear_narx"]):
        model_offset = 0.015 * model_idx
        for split_idx, split in enumerate(splits):
            for i in range(n_per_split):
                if split == "id_test":
                    rmse = 0.06 + 0.008 * i + model_offset
                    support = 0.10 + 0.01 * i
                    disturbance_scale = 0.9
                elif split == "ood_action_magnitude":
                    rmse = 0.12 + 0.025 * i + model_offset
                    support = 0.75 + 0.05 * i
                    disturbance_scale = 1.0
                elif split == "ood_inflow_spike":
                    rmse = 0.10 + 0.022 * i + model_offset
                    support = 0.35 + 0.03 * i
                    disturbance_scale = 2.8
                else:
                    rmse = 0.16 + 0.030 * i + model_offset
                    support = 0.90 + 0.04 * i
                    disturbance_scale = 3.0
                uncertainty = 0.05 + rmse * 0.7 + 0.01 * model_idx
                disagreement = 0.04 + 0.12 * split_idx + 0.005 * i
                invariant = max(0.0, rmse - 0.08) * 0.5
                repair = 0.03 * float(rmse > threshold) + 0.005 * split_idx
                combined = support + uncertainty + disagreement + invariant + repair
                rows.append(
                    {
                        "scenario_id": f"{split}_{i}",
                        "system_id": "two_tank",
                        "split": split,
                        "scenario_type": scenario_types[split],
                        "model_id": model_id,
                        "error": rmse,
                        "mae": rmse * 0.8,
                        "max_abs_error": rmse * 1.3,
                        "final_state_error": rmse * 0.9,
                        "support_distance": support,
                        "uncertainty": uncertainty,
                        "disagreement": disagreement,
                        "invariant_residual": invariant,
                        "repair_amount": repair,
                        "risk_support_only": support,
                        "risk_uncertainty_only": uncertainty,
                        "risk_disagreement_only": disagreement,
                        "risk_invariant_only": invariant,
                        "risk_repair_only": repair,
                        "risk_combined_linear": combined,
                        "risk_random_baseline": float(rng.uniform(0.0, 1.0)),
                        "risk_oracle_error_rank": rmse,
                        "disturbance_scale": disturbance_scale,
                    }
                )
    scenario = pd.DataFrame(rows)
    scenario.drop(columns=["disturbance_scale"]).to_csv(path / "scenario_scores.csv", index=False)

    risk_rows = []
    for model_id in scenario["model_id"].unique():
        for split in scenario["split"].unique():
            for judge in PLOT_JUDGES:
                for coverage in [0.25, 0.50, 1.00]:
                    risk_rows.append(
                        {
                            "system_id": "two_tank",
                            "model_id": model_id,
                            "split": split,
                            "judge_id": judge,
                            "coverage": coverage,
                            "false_accept_rate": 0.25,
                            "accepted_count": 1,
                            "false_accept_count": 0,
                            "mean_error_accepted": 0.1,
                            "mean_error_rejected": 0.2,
                            "threshold": threshold,
                        }
                    )
    pd.DataFrame(risk_rows).to_csv(path / "risk_coverage.csv", index=False)
    pd.DataFrame(
        {
            "system_id": ["two_tank", "two_tank"],
            "model_id": ["hold_last", "linear_narx"],
            "split": ["id_test", "id_test"],
            "rmse_mean": [0.12, 0.08],
        }
    ).to_csv(path / "model_metrics.csv", index=False)
    summary = {
        "dataset_summary": {
            "train": {
                "action_min": 0.0,
                "action_max": 1.0,
                "disturbance_0_max": 1.0,
            },
            "id_test": {
                "action_min": 0.0,
                "action_max": 1.0,
                "disturbance_0_max": 1.0,
            },
            "ood_action_magnitude": {
                "action_min": 0.0,
                "action_max": 3.0,
                "disturbance_0_max": 1.0,
            },
            "ood_inflow_spike": {
                "action_min": 0.0,
                "action_max": 1.0,
                "disturbance_0_max": 3.2,
            },
            "ood_combined": {
                "action_min": 0.0,
                "action_max": 3.0,
                "disturbance_0_max": 3.3,
            },
        }
    }
    (path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return path


@pytest.fixture()
def failure_results(tmp_path: Path) -> Path:
    return write_failure_results_fixture(tmp_path / "results")


@pytest.fixture()
def failure_table_path(tmp_path: Path, failure_results: Path) -> Path:
    output = tmp_path / "failure_analysis" / "failure_table.csv"
    build_failure_table(failure_results, output)
    return output


def write_tiny_calibrated_config(path: Path, seed: int = 11) -> Path:
    config = {
        "experiment_id": "calibrated_two_tank_tiny",
        "seed": seed,
        "system_id": "two_tank",
        "horizon": 12,
        "dt": 0.1,
        "n_model_train": 24,
        "n_calibration_id": 8,
        "n_calibration_ood": 8,
        "n_test_id": 10,
        "n_test_ood": 10,
        "output_dir": str(path.parent / "calibrated"),
        "uncertainty_samples": 1,
        "models": ["hold_last", "linear_narx"],
        "signals": [
            "support_distance",
            "uncertainty_score",
            "disagreement_score",
            "invariant_residual",
            "repair_amount",
        ],
        "judges": [
            "support_only",
            "uncertainty_only",
            "disagreement_only",
            "invariant_only",
            "repair_only",
            "combined_linear",
            "best_single_signal_selected_on_calibration",
            "rank_normalized_linear",
            "calibration_selected_candidate_ranker",
            "logistic_calibrated_judge",
            "isotonic_calibrated_judge",
            "quantile_rule_judge",
            "conservative_low_coverage_judge",
            "random_baseline",
            "oracle_error_rank",
        ],
        "bad_threshold": {"metric": "rmse", "value": 0.15},
        "coverages": [0.05, 0.10, 0.20, 0.40, 1.00],
        "primary_coverages": [0.05, 0.10],
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def write_tiny_calibrated_cstr_config(path: Path, seed: int = 11) -> Path:
    config = {
        "experiment_id": "calibrated_cstr_tiny",
        "seed": seed,
        "system_id": "cstr",
        "horizon": 12,
        "dt": 0.1,
        "n_model_train": 24,
        "n_calibration_id": 8,
        "n_calibration_ood": 8,
        "n_test_id": 10,
        "n_test_ood": 10,
        "output_dir": str(path.parent / "calibrated_cstr"),
        "uncertainty_samples": 1,
        "models": ["hold_last", "linear_narx"],
        "signals": [
            "support_distance",
            "uncertainty_score",
            "disagreement_score",
            "invariant_residual",
            "repair_amount",
        ],
        "judges": [
            "support_only",
            "uncertainty_only",
            "disagreement_only",
            "invariant_only",
            "repair_only",
            "combined_linear",
            "best_single_signal_selected_on_calibration",
            "rank_normalized_linear",
            "calibration_selected_candidate_ranker",
            "logistic_calibrated_judge",
            "isotonic_calibrated_judge",
            "quantile_rule_judge",
            "conservative_low_coverage_judge",
            "random_baseline",
            "oracle_error_rank",
        ],
        "bad_threshold": {"metric": "rmse", "value": 0.15},
        "coverages": [0.05, 0.10, 0.20, 0.40, 1.00],
        "primary_coverages": [0.05, 0.10],
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path
