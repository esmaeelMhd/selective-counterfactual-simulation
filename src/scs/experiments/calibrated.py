from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from scs.data.generate import generate_calibrated_two_tank_dataset, summarize_dataset
from scs.data.schemas import save_dataset
from scs.experiments.registry import make_model, make_system
from scs.metrics.trajectory import final_state_error, mae, max_abs_error, rmse
from scs.validators.calibrated import CALIBRATED_JUDGE_IDS, make_calibrated_judges
from scs.validators.disagreement import disagreement_score
from scs.validators.invariants import invariant_residual_score
from scs.validators.repair import repair_amount_score
from scs.validators.support import SupportDistance
from scs.validators.uncertainty import uncertainty_score


SIMPLE_JUDGES = [
    "support_only",
    "uncertainty_only",
    "disagreement_only",
    "invariant_only",
    "repair_only",
]
EXISTING_JUDGES = [*SIMPLE_JUDGES, "combined_linear", "random_baseline", "oracle_error_rank"]
REAL_JUDGES = [*SIMPLE_JUDGES, "combined_linear", *CALIBRATED_JUDGE_IDS, "random_baseline"]
ORACLE_JUDGE = "oracle_error_rank"
OLD_REPO_NAMES = [
    "time" + "-series" + "-simulator",
    "digital" + "-twin" + "-engine",
    "flux" + "-attention" + "-engine",
    "plant" + "-scenario" + "-compiler",
]
SIGNAL_COLUMN_BY_SIMPLE_JUDGE = {
    "support_only": "support_distance",
    "uncertainty_only": "uncertainty_score",
    "disagreement_only": "disagreement_score",
    "invariant_only": "invariant_residual",
    "repair_only": "repair_amount",
}
REQUIRED_CONFIG_KEYS = {
    "experiment_id",
    "seed",
    "system_id",
    "horizon",
    "dt",
    "n_model_train",
    "n_calibration_id",
    "n_calibration_ood",
    "n_test_id",
    "n_test_ood",
    "models",
    "signals",
    "judges",
    "bad_threshold",
    "coverages",
    "primary_coverages",
}
SPLIT_SEED_OFFSETS = {
    "model_train": 1001,
    "judge_calibration_id": 2001,
    "judge_calibration_ood_action_magnitude": 2101,
    "judge_calibration_ood_inflow_spike": 2201,
    "judge_calibration_ood_combined": 2301,
    "judge_calibration_pump_degradation": 2401,
    "judge_test_id": 3001,
    "judge_test_ood_action_magnitude": 3101,
    "judge_test_ood_inflow_spike": 3201,
    "judge_test_ood_combined": 3301,
    "judge_test_pump_degradation": 3401,
}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    table = df[columns].copy()
    if max_rows is not None:
        table = table.head(max_rows)
    if table.empty:
        return "| " + " | ".join(columns) + " |\n| " + " | ".join(["---"] * len(columns)) + " |"
    lines = ["| " + " | ".join(columns) + " |"]
    lines.append("| " + " | ".join("---:" if pd.api.types.is_numeric_dtype(table[col]) else "---" for col in columns) + " |")
    for _, row in table.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append("nan" if pd.isna(value) else f"{float(value):.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def load_calibrated_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("calibrated config must be a mapping")
    missing = sorted(REQUIRED_CONFIG_KEYS - set(config))
    if missing:
        raise ValueError(f"missing calibrated config keys: {missing}")
    if config["system_id"] != "two_tank":
        raise ValueError("calibrated judge evidence is restricted to system_id='two_tank'")
    forbidden = {"cstr", "heat_exchanger", "rssm"}
    serialized = yaml.safe_dump(config).lower()
    used_forbidden = sorted(item for item in forbidden if re.search(rf"\b{re.escape(item)}\b", serialized))
    if used_forbidden:
        raise ValueError(f"forbidden calibrated evidence entries in config: {used_forbidden}")
    return config


def _trajectory_hash(states: np.ndarray, actions: np.ndarray, disturbances: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(states).tobytes())
    digest.update(np.ascontiguousarray(actions).tobytes())
    digest.update(np.ascontiguousarray(disturbances).tobytes())
    return digest.hexdigest()


def _split_role(split: str) -> str:
    if split == "model_train":
        return "model_train"
    if split.startswith("judge_calibration"):
        return "judge_calibration"
    if split.startswith("judge_test"):
        return "judge_test"
    raise ValueError(f"unknown calibrated split role: {split}")


def _split_group(split: str) -> str:
    if split.endswith("_id"):
        return "normal_policy"
    if "action_magnitude" in split:
        return "held_out_action_magnitude"
    if "inflow_spike" in split:
        return "inflow_spike"
    if "combined" in split:
        return "combined_intervention"
    if "pump_degradation" in split:
        return "valve_or_pump_degradation"
    if split == "model_train":
        return "normal_policy"
    return split


def _build_split_summary(config: dict[str, Any], dataset: dict) -> pd.DataFrame:
    rows = []
    for split, batch in dataset.items():
        hashes = [
            _trajectory_hash(batch.states[i], batch.actions[i], batch.disturbances[i])
            for i in range(batch.n_trajectories)
        ]
        rows.append(
            {
                "split": split,
                "role": _split_role(split),
                "split_group": _split_group(split),
                "scenario_type": ",".join(sorted(set(batch.scenario_type))),
                "n_trajectories": batch.n_trajectories,
                "action_min": float(np.min(batch.actions)),
                "action_max": float(np.max(batch.actions)),
                "disturbance_0_max": float(np.max(batch.disturbances[..., 0])),
                "disturbance_1_max": float(np.max(batch.disturbances[..., 1])) if batch.disturbances.shape[-1] > 1 else 0.0,
                "seed_stream": int(config["seed"]) + SPLIT_SEED_OFFSETS[split],
                "trajectory_hash_set": hashlib.sha256("\n".join(sorted(hashes)).encode("utf-8")).hexdigest(),
            }
        )
    return pd.DataFrame(rows).sort_values(["role", "split"])


def _validate_calibrated_splits(dataset: dict, split_summary: pd.DataFrame) -> dict[str, Any]:
    scenario_ids: dict[str, set[str]] = {}
    trajectory_hashes: dict[str, set[str]] = {}
    for split, batch in dataset.items():
        scenario_ids[split] = {f"{split}_{i:04d}" for i in range(batch.n_trajectories)}
        trajectory_hashes[split] = {
            _trajectory_hash(batch.states[i], batch.actions[i], batch.disturbances[i])
            for i in range(batch.n_trajectories)
        }
    calibration_ids = set().union(*(scenario_ids[name] for name in scenario_ids if name.startswith("judge_calibration")))
    test_ids = set().union(*(scenario_ids[name] for name in scenario_ids if name.startswith("judge_test")))
    calibration_hashes = set().union(*(trajectory_hashes[name] for name in trajectory_hashes if name.startswith("judge_calibration")))
    test_hashes = set().union(*(trajectory_hashes[name] for name in trajectory_hashes if name.startswith("judge_test")))
    id_action_range = float(
        split_summary.loc[split_summary["split"] == "judge_calibration_id", "action_max"].iloc[0]
        - split_summary.loc[split_summary["split"] == "judge_calibration_id", "action_min"].iloc[0]
    )
    ood_action_range = float(
        split_summary.loc[split_summary["split"] == "judge_calibration_ood_action_magnitude", "action_max"].iloc[0]
        - split_summary.loc[split_summary["split"] == "judge_calibration_ood_action_magnitude", "action_min"].iloc[0]
    )
    id_inflow = float(split_summary.loc[split_summary["split"] == "judge_calibration_id", "disturbance_0_max"].iloc[0])
    ood_inflow = float(split_summary.loc[split_summary["split"] == "judge_calibration_ood_inflow_spike", "disturbance_0_max"].iloc[0])
    result = {
        "scenario_overlap_count": len(calibration_ids & test_ids),
        "trajectory_overlap_count": len(calibration_hashes & test_hashes),
        "seed_streams_unique": bool(split_summary["seed_stream"].nunique() == len(split_summary)),
        "ood_action_range_ratio": ood_action_range / max(id_action_range, 1e-12),
        "ood_inflow_ratio": ood_inflow / max(id_inflow, 1e-12),
    }
    if result["scenario_overlap_count"] != 0:
        raise RuntimeError("calibration/test scenario IDs overlap")
    if result["trajectory_overlap_count"] != 0:
        raise RuntimeError("calibration/test trajectories overlap")
    if not result["seed_streams_unique"]:
        raise RuntimeError("calibrated split seed streams are not unique")
    if result["ood_action_range_ratio"] <= 1.25:
        raise RuntimeError("calibration OOD action split is not measurably different from ID")
    if result["ood_inflow_ratio"] <= 1.25:
        raise RuntimeError("calibration OOD inflow split is not measurably different from ID")
    return result


def generate_calibrated_data(config: dict[str, Any], output_dir: str | Path) -> dict:
    out_dir = Path(output_dir)
    dataset = generate_calibrated_two_tank_dataset(
        n_model_train=int(config["n_model_train"]),
        n_calibration_id=int(config["n_calibration_id"]),
        n_calibration_ood=int(config["n_calibration_ood"]),
        n_test_id=int(config["n_test_id"]),
        n_test_ood=int(config["n_test_ood"]),
        horizon=int(config["horizon"]),
        dt=float(config["dt"]),
        seed=int(config["seed"]),
        severity=str(config.get("severity", "medium")),
    )
    _ensure_dir(out_dir)
    save_dataset(dataset, out_dir / "data")
    data_summary = summarize_dataset(dataset)
    _write_json(out_dir / "data_summary.json", data_summary)
    split_summary = _build_split_summary(config, dataset)
    split_summary.to_csv(out_dir / "split_summary.csv", index=False)
    integrity = _validate_calibrated_splits(dataset, split_summary)
    _write_json(out_dir / "split_integrity.json", integrity)
    return dataset


def _score_scenarios(config: dict[str, Any], dataset: dict, role: str, models: list, support: SupportDistance) -> pd.DataFrame:
    system = make_system("two_tank")
    split_names = [name for name in dataset if _split_role(name) == role]
    predictions: dict[str, dict[tuple[str, int], np.ndarray]] = {model.model_id: {} for model in models}
    true_by_key: dict[tuple[str, int], np.ndarray] = {}
    actions_by_key: dict[tuple[str, int], np.ndarray] = {}
    disturbances_by_key: dict[tuple[str, int], np.ndarray] = {}
    scenario_type_by_key: dict[tuple[str, int], str] = {}
    for split in split_names:
        batch = dataset[split]
        for i in range(batch.n_trajectories):
            key = (split, i)
            true_by_key[key] = batch.states[i]
            actions_by_key[key] = batch.actions[i]
            disturbances_by_key[key] = batch.disturbances[i]
            scenario_type_by_key[key] = batch.scenario_type[i]
            for model in models:
                predictions[model.model_id][key] = model.predict_rollout(
                    batch.states[i, 0],
                    batch.actions[i],
                    batch.disturbances[i],
                )

    rows = []
    n_samples = int(config.get("uncertainty_samples", 3))
    for key in sorted(true_by_key):
        split, i = key
        all_predictions = [predictions[model.model_id][key] for model in models]
        scenario_disagreement = disagreement_score(all_predictions)
        for model in models:
            predicted = predictions[model.model_id][key]
            actual = true_by_key[key]
            scenario_id = f"{split}_{i:04d}"
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "system_id": system.system_id,
                    "role": role,
                    "split": split,
                    "split_group": _split_group(split),
                    "scenario_type": scenario_type_by_key[key],
                    "model_id": model.model_id,
                    "rmse": rmse(predicted, actual),
                    "mae": mae(predicted, actual),
                    "max_abs_error": max_abs_error(predicted, actual),
                    "final_state_error": final_state_error(predicted, actual),
                    "support_distance": support.score(actions_by_key[key], disturbances_by_key[key]),
                    "uncertainty_score": uncertainty_score(
                        model,
                        actual[0],
                        actions_by_key[key],
                        disturbances_by_key[key],
                        n_samples=n_samples,
                    ),
                    "disagreement_score": scenario_disagreement,
                    "invariant_residual": invariant_residual_score(
                        system,
                        predicted,
                        actions_by_key[key],
                        disturbances_by_key[key],
                        float(config["dt"]),
                    ),
                    "repair_amount": repair_amount_score(system, predicted),
                }
            )
    table = pd.DataFrame(rows)
    if table.empty:
        raise RuntimeError(f"{role} table is empty")
    if not np.isfinite(table.select_dtypes(include=[float, int]).to_numpy()).all():
        raise RuntimeError(f"{role} table contains non-finite values")
    return table


def _fit_existing_score_context(calibration_table: pd.DataFrame, signal_columns: list[str]) -> dict[str, dict[str, float]]:
    context = {}
    for signal in signal_columns:
        values = calibration_table[signal].astype(float)
        context[signal] = {"min": float(values.min()), "max": float(values.max())}
    return context


def _normalize_with_context(values: pd.Series, context: dict[str, float]) -> pd.Series:
    denom = context["max"] - context["min"]
    if denom <= 1e-12:
        return pd.Series(np.zeros(len(values)), index=values.index, dtype=float)
    return (values.astype(float) - context["min"]) / denom


def _existing_judge_scores(
    table: pd.DataFrame,
    judge_ids: list[str],
    signal_columns: list[str],
    normalizer: dict[str, dict[str, float]],
    seed: int,
) -> pd.DataFrame:
    scores = pd.DataFrame(index=table.index)
    for judge_id, signal in SIGNAL_COLUMN_BY_SIMPLE_JUDGE.items():
        if judge_id in judge_ids:
            scores[judge_id] = table[signal].astype(float)
    if "combined_linear" in judge_ids:
        normalized = [_normalize_with_context(table[signal], normalizer[signal]) for signal in signal_columns]
        scores["combined_linear"] = sum(normalized) / len(normalized)
    if "random_baseline" in judge_ids:
        rng = np.random.default_rng(seed)
        scores["random_baseline"] = rng.uniform(0.0, 1.0, size=len(table))
    if "oracle_error_rank" in judge_ids:
        scores["oracle_error_rank"] = table["rmse"].astype(float)
    return scores


def _risk_rows_for_group(
    group: pd.DataFrame,
    judge_id: str,
    score_column: str,
    coverages: list[float],
    threshold: float,
    is_real_judge: bool,
    is_oracle: bool,
    is_calibrated: bool,
) -> list[dict[str, Any]]:
    scores = group[score_column].to_numpy(dtype=float)
    errors = group["rmse"].to_numpy(dtype=float)
    order = np.argsort(scores, kind="mergesort")
    rows = []
    for coverage in coverages:
        accepted_count = min(max(int(np.ceil(float(coverage) * len(group))), 1), len(group))
        accepted_idx = order[:accepted_count]
        rejected_idx = order[accepted_count:]
        accepted_errors = errors[accepted_idx]
        false_accept_count = int(np.sum(accepted_errors > threshold))
        rows.append(
            {
                "system_id": str(group["system_id"].iloc[0]),
                "model_id": str(group["model_id"].iloc[0]),
                "judge_id": judge_id,
                "split_group": str(group["split_group"].iloc[0]),
                "scenario_type": str(group["scenario_type"].iloc[0]),
                "coverage_requested": float(coverage),
                "coverage_achieved": float(accepted_count / len(group)),
                "false_accept_rate": float(false_accept_count / accepted_count),
                "accepted_count": accepted_count,
                "false_accept_count": false_accept_count,
                "bad_count_total": int(np.sum(errors > threshold)),
                "mean_error_accepted": float(np.mean(accepted_errors)),
                "mean_error_rejected": float(np.mean(errors[rejected_idx])) if len(rejected_idx) else 0.0,
                "threshold": float(threshold),
                "is_real_judge": bool(is_real_judge),
                "is_oracle": bool(is_oracle),
                "is_calibrated": bool(is_calibrated),
            }
        )
    return rows


def _compute_risk_coverage(scored_test: pd.DataFrame, judge_ids: list[str], coverages: list[float], threshold: float) -> pd.DataFrame:
    rows = []
    for (_, _), group in scored_test.groupby(["model_id", "scenario_type"], sort=False):
        for judge_id in judge_ids:
            score_column = f"risk_{judge_id}"
            rows.extend(
                _risk_rows_for_group(
                    group=group,
                    judge_id=judge_id,
                    score_column=score_column,
                    coverages=coverages,
                    threshold=threshold,
                    is_real_judge=judge_id != ORACLE_JUDGE,
                    is_oracle=judge_id == ORACLE_JUDGE,
                    is_calibrated=judge_id in CALIBRATED_JUDGE_IDS,
                )
            )
    result = pd.DataFrame(rows)
    numeric = result.select_dtypes(include=[float, int])
    if result.empty or not np.isfinite(numeric.to_numpy()).all():
        raise RuntimeError("calibrated_risk_coverage is empty or non-finite")
    return result


def _plot_calibrated_risk(risk: pd.DataFrame, output: Path) -> None:
    _ensure_dir(output.parent)
    plot_df = risk.groupby(["judge_id", "coverage_requested"], as_index=False)["false_accept_rate"].mean()
    judges = [
        "best_single_signal_selected_on_calibration",
        "rank_normalized_linear",
        "logistic_calibrated_judge",
        "isotonic_calibrated_judge",
        "quantile_rule_judge",
        "conservative_low_coverage_judge",
        "combined_linear",
        "random_baseline",
        "oracle_error_rank",
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    for judge in judges:
        frame = plot_df[plot_df["judge_id"] == judge].sort_values("coverage_requested")
        if frame.empty:
            continue
        label = f"{judge} (diagnostic)" if judge == ORACLE_JUDGE else judge
        ax.plot(frame["coverage_requested"], frame["false_accept_rate"], marker="o", linewidth=1.5, label=label)
    ax.set_xlabel("coverage")
    ax.set_ylabel("false_accept_rate")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _build_comparison(risk: pd.DataFrame, primary_coverages: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    calibrated_candidates = [judge for judge in CALIBRATED_JUDGE_IDS if judge != "best_single_signal_selected_on_calibration"]
    for (model_id, scenario_type, coverage), group in risk.groupby(["model_id", "scenario_type", "coverage_requested"], sort=False):
        baseline = group[group["judge_id"] == "best_single_signal_selected_on_calibration"]
        if baseline.empty:
            continue
        baseline_far = float(baseline["false_accept_rate"].iloc[0])
        combined_far = float(group.loc[group["judge_id"] == "combined_linear", "false_accept_rate"].iloc[0])
        random_far = float(group.loc[group["judge_id"] == "random_baseline", "false_accept_rate"].iloc[0])
        calibrated = group[group["judge_id"].isin(calibrated_candidates)].sort_values("false_accept_rate")
        best_calibrated = calibrated.iloc[0]
        simple = group[group["judge_id"].isin(SIMPLE_JUDGES)].sort_values("false_accept_rate")
        best_simple = simple.iloc[0]
        rows.append(
            {
                "model_id": model_id,
                "scenario_type": scenario_type,
                "coverage": float(coverage),
                "best_deployable_baseline": "best_single_signal_selected_on_calibration",
                "baseline_far": baseline_far,
                "best_calibrated_judge": str(best_calibrated["judge_id"]),
                "calibrated_far": float(best_calibrated["false_accept_rate"]),
                "margin": baseline_far - float(best_calibrated["false_accept_rate"]),
                "beats_baseline": bool(float(best_calibrated["false_accept_rate"]) < baseline_far - 1e-12),
                "combined_linear_far": combined_far,
                "beats_combined_linear": bool(float(best_calibrated["false_accept_rate"]) < combined_far - 1e-12),
                "random_baseline_far": random_far,
                "best_simple_real_judge_on_test": str(best_simple["judge_id"]),
                "best_simple_real_judge_on_test_far": float(best_simple["false_accept_rate"]),
            }
        )
    comparison = pd.DataFrame(rows)
    low_rows = []
    for coverage, group in comparison[comparison["coverage"].isin(primary_coverages)].groupby("coverage", sort=True):
        idx = group["calibrated_far"].idxmin()
        best = group.loc[idx]
        low_rows.append(
            {
                "coverage": float(coverage),
                "best_deployable_baseline": "best_single_signal_selected_on_calibration",
                "baseline_far": float(group["baseline_far"].mean()),
                "best_calibrated_judge": str(best["best_calibrated_judge"]),
                "calibrated_far": float(group["calibrated_far"].mean()),
                "margin": float(group["baseline_far"].mean() - group["calibrated_far"].mean()),
                "win_rate_vs_baseline": float(group["beats_baseline"].mean()),
                "win_rate_vs_combined_linear": float(group["beats_combined_linear"].mean()),
            }
        )
    return comparison, pd.DataFrame(low_rows)


def _verdict_from_low_coverage(low: pd.DataFrame, comparison: pd.DataFrame, leakage_detected: bool) -> str:
    if leakage_detected:
        return "INVALID_DUE_TO_LEAKAGE"
    if low.empty:
        return "NO_IMPROVEMENT_OVER_SINGLE_SIGNAL"
    supported = low[(low["margin"] > 1e-12) & (low["win_rate_vs_combined_linear"] > 0.0)]
    if not supported.empty:
        return "SUPPORTED_LOW_COVERAGE"
    if bool(comparison["beats_baseline"].any()):
        return "MIXED"
    return "NO_IMPROVEMENT_OVER_SINGLE_SIGNAL"


def evaluate_calibrated_tables(
    config: dict[str, Any],
    calibration_table: pd.DataFrame,
    test_table: pd.DataFrame,
    output_dir: str | Path,
    report_path: str | Path | None,
    command: str,
) -> dict[str, Any]:
    out_dir = Path(output_dir)
    _ensure_dir(out_dir)
    signal_columns = [str(column) for column in config["signals"]]
    threshold = float(config["bad_threshold"]["value"])
    coverages = [float(value) for value in config["coverages"]]
    primary_coverages = [float(value) for value in config["primary_coverages"]]
    calibration = calibration_table.copy()
    test = test_table.copy()
    calibration["bad_rmse_label"] = calibration["rmse"].astype(float) > threshold
    test["bad_rmse_label"] = test["rmse"].astype(float) > threshold

    overlap = set(calibration["scenario_id"]) & set(test["scenario_id"])
    leakage_detected = bool(overlap)
    if leakage_detected:
        raise RuntimeError("calibration/test scenario IDs overlap")

    normalizer = _fit_existing_score_context(calibration, signal_columns)
    judge_ids = [str(judge) for judge in config["judges"]]
    existing = [judge for judge in judge_ids if judge in EXISTING_JUDGES]
    calibrated_ids = [judge for judge in judge_ids if judge in CALIBRATED_JUDGE_IDS]

    scored_calibration = calibration.copy()
    scored_test = test.copy()
    existing_cal = _existing_judge_scores(scored_calibration, existing, signal_columns, normalizer, int(config["seed"]) + 7001)
    existing_test = _existing_judge_scores(scored_test, existing, signal_columns, normalizer, int(config["seed"]) + 8001)
    for judge in existing_cal.columns:
        scored_calibration[f"risk_{judge}"] = existing_cal[judge].to_numpy(dtype=float)
        scored_test[f"risk_{judge}"] = existing_test[judge].to_numpy(dtype=float)

    calibrated_judges = make_calibrated_judges(primary_coverages, threshold)
    calibrated_judges = [judge for judge in calibrated_judges if judge.judge_id in calibrated_ids]
    for judge in calibrated_judges:
        judge.fit(scored_calibration, signal_columns, "rmse", "bad_rmse_label")
        judge.set_test_scenario_hash(scored_test["scenario_id"])
        scored_calibration[f"risk_{judge.judge_id}"] = judge.score(scored_calibration)
        scored_test[f"risk_{judge.judge_id}"] = judge.score(scored_test)

    provenance = [judge.provenance() for judge in calibrated_judges]
    leakage_detected = leakage_detected or any(item["used_test_labels_during_fit"] for item in provenance)
    if leakage_detected:
        verdict = "INVALID_DUE_TO_LEAKAGE"
    if not np.isfinite(scored_test[[f"risk_{judge}" for judge in judge_ids]].to_numpy(dtype=float)).all():
        raise RuntimeError("calibrated test risk scores contain non-finite values")

    scored_calibration.to_csv(out_dir / "calibration_table.csv", index=False)
    scored_test.to_csv(out_dir / "test_table.csv", index=False)
    _write_json(out_dir / "judge_provenance.json", {"judges": provenance})
    selection = pd.DataFrame(
        [
            {
                "judge_id": item["judge_id"],
                "available": item["available"],
                "unavailable_reason": item["unavailable_reason"],
                "selected_signal_if_any": item["selected_signal_if_any"],
                "selected_hyperparameters": json.dumps(item["selected_hyperparameters"], sort_keys=True),
                "used_test_labels_during_fit": item["used_test_labels_during_fit"],
            }
            for item in provenance
        ]
    )
    selection.to_csv(out_dir / "calibration_selection.csv", index=False)
    risk = _compute_risk_coverage(scored_test, judge_ids, coverages, threshold)
    risk.to_csv(out_dir / "calibrated_risk_coverage.csv", index=False)
    _plot_calibrated_risk(risk, out_dir / "calibrated_risk_coverage.png")
    comparison, low = _build_comparison(risk, primary_coverages)
    comparison.to_csv(out_dir / "test_comparison.csv", index=False)
    low.to_csv(out_dir / "low_coverage_summary.csv", index=False)
    verdict = _verdict_from_low_coverage(low, comparison, leakage_detected)
    best_row = low.sort_values("margin", ascending=False).iloc[0] if not low.empty else None
    summary = {
        "verdict": verdict,
        "best_calibrated_judge": None if best_row is None else str(best_row["best_calibrated_judge"]),
        "best_calibration_selected_single_signal": "best_single_signal_selected_on_calibration",
        "low_coverage_win": bool(not low.empty and (low["margin"] > 1e-12).any()),
        "low_coverage_win_rate": float(low["win_rate_vs_baseline"].mean()) if not low.empty else 0.0,
        "leakage_detected": bool(leakage_detected),
        "unavailable_judges": [item["judge_id"] for item in provenance if not item["available"]],
        "oracle_is_diagnostic_only": True,
        "expansion_allowed": False,
        "artifacts": {
            "calibration_table": str(out_dir / "calibration_table.csv"),
            "test_table": str(out_dir / "test_table.csv"),
            "judge_provenance": str(out_dir / "judge_provenance.json"),
            "calibration_selection": str(out_dir / "calibration_selection.csv"),
            "calibrated_risk_coverage": str(out_dir / "calibrated_risk_coverage.csv"),
            "test_comparison": str(out_dir / "test_comparison.csv"),
            "low_coverage_summary": str(out_dir / "low_coverage_summary.csv"),
            "calibrated_risk_coverage_plot": str(out_dir / "calibrated_risk_coverage.png"),
        },
    }
    _write_json(out_dir / "calibrated_judge_summary.json", summary)
    if report_path is not None:
        write_calibrated_report(config, summary, risk, comparison, low, selection, Path(report_path), command)
    return summary


def run_calibrated_judge(
    config_path: str | Path,
    output: str | Path,
    report_path: str | Path | None = "reports/calibrated_refusal_judge.md",
    command: str | None = None,
) -> dict[str, Any]:
    config = load_calibrated_config(config_path)
    out_dir = Path(output)
    command = command or f"python scripts/run_calibrated_judge.py --config {config_path} --output {output}"
    np.random.seed(int(config["seed"]))
    dataset = generate_calibrated_data(config, out_dir)
    support = SupportDistance()
    support.fit(dataset["model_train"])
    models = [make_model(str(model_id), seed=int(config["seed"]) + idx) for idx, model_id in enumerate(config["models"])]
    for model in models:
        model.fit(dataset["model_train"])
    calibration_table = _score_scenarios(config, dataset, "judge_calibration", models, support)
    test_table = _score_scenarios(config, dataset, "judge_test", models, support)
    summary = evaluate_calibrated_tables(config, calibration_table, test_table, out_dir, report_path, command)
    required = [
        out_dir / "data_summary.json",
        out_dir / "split_summary.csv",
        out_dir / "calibration_table.csv",
        out_dir / "test_table.csv",
        out_dir / "judge_provenance.json",
        out_dir / "calibration_selection.csv",
        out_dir / "calibrated_risk_coverage.csv",
        out_dir / "test_comparison.csv",
        out_dir / "low_coverage_summary.csv",
        out_dir / "calibrated_judge_summary.json",
        out_dir / "calibrated_risk_coverage.png",
    ]
    if report_path is not None:
        required.append(Path(report_path))
    missing = [str(path) for path in required if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"missing calibrated artifacts: {missing}")
    return summary


def write_calibrated_report(
    config: dict[str, Any],
    summary: dict[str, Any],
    risk: pd.DataFrame,
    comparison: pd.DataFrame,
    low: pd.DataFrame,
    selection: pd.DataFrame,
    output: Path,
    command: str,
) -> None:
    _ensure_dir(output.parent)
    aggregate = risk.groupby(["judge_id", "coverage_requested"], as_index=False)["false_accept_rate"].mean()
    oracle = aggregate[aggregate["judge_id"] == ORACLE_JUDGE].rename(columns={"false_accept_rate": "oracle_far"})
    best_real = (
        aggregate[aggregate["judge_id"].isin(REAL_JUDGES)]
        .groupby("coverage_requested", as_index=False)["false_accept_rate"]
        .min()
        .rename(columns={"false_accept_rate": "best_real_far"})
    )
    oracle_gap = oracle.merge(best_real, on="coverage_requested", how="inner")
    oracle_gap["oracle_gap"] = oracle_gap["best_real_far"] - oracle_gap["oracle_far"]
    unavailable = selection[~selection["available"]]
    text = f"""# Calibrated Refusal Judge Report

## Research question

Can a calibrated refusal judge identify a low-coverage subset of counterfactual rollouts with lower false-accept risk than the strongest calibration-selected single signal?

## Strict leakage statement

Simulator models are fit on `model_train`; calibrated judges are selected or fit on `judge_calibration_*`; final risk-coverage is computed on `judge_test_*`. Test labels are not used during judge fitting. Oracle is diagnostic only.

## Data splits

model_train, judge_calibration_id, judge_calibration_ood_action_magnitude, judge_calibration_ood_inflow_spike, judge_calibration_ood_combined, judge_calibration_pump_degradation, judge_test_id, judge_test_ood_action_magnitude, judge_test_ood_inflow_spike, judge_test_ood_combined, judge_test_pump_degradation.

## Models

{", ".join(config["models"])}

## Signals

{", ".join(config["signals"])}

## Judges

{", ".join(config["judges"])}

## Calibration provenance

{_markdown_table(selection, ["judge_id", "available", "selected_signal_if_any", "used_test_labels_during_fit"])}

## Low-coverage result

{_markdown_table(low, ["coverage", "best_deployable_baseline", "baseline_far", "best_calibrated_judge", "calibrated_far", "margin"])}

## Full risk-coverage result

{_markdown_table(aggregate, ["judge_id", "coverage_requested", "false_accept_rate"], max_rows=80)}

## Comparison against v0 combined_linear

{_markdown_table(comparison, ["model_id", "scenario_type", "coverage", "best_calibrated_judge", "calibrated_far", "combined_linear_far", "beats_combined_linear"], max_rows=30)}

## Comparison against best single signal selected on calibration

{_markdown_table(comparison, ["model_id", "scenario_type", "coverage", "best_deployable_baseline", "baseline_far", "best_calibrated_judge", "calibrated_far", "margin"], max_rows=30)}

## Oracle diagnostic gap

{_markdown_table(oracle_gap, ["coverage_requested", "oracle_far", "best_real_far", "oracle_gap"])}

## Unavailable judges

{_markdown_table(unavailable, ["judge_id", "unavailable_reason"]) if not unavailable.empty else "none"}

## Verdict

{summary["verdict"]}

## Explanation

Best calibrated judge: {summary["best_calibrated_judge"]}. Low-coverage win rate versus the calibration-selected single-signal baseline: {summary["low_coverage_win_rate"]:.6f}. Expansion remains forbidden.

## Known failures

{"- none" if summary["verdict"] == "SUPPORTED_LOW_COVERAGE" else "- calibrated judges did not establish a broad selective-simulation claim"}

## Reproduction command

```bash
{command}
```
"""
    output.write_text(text, encoding="utf-8")


def verify_calibrated_preconditions(config_path: str | Path = "configs/experiments/calibrated_two_tank.yaml") -> dict[str, Any]:
    out_dir = Path("results/calibrated_two_tank")
    _ensure_dir(out_dir)
    report_path = Path("reports/calibrated_precondition_check.md")
    diagnosis_path = Path("reports/failure_diagnosis.json")
    if not diagnosis_path.exists():
        raise FileNotFoundError(diagnosis_path)
    diagnosis = _load_json(diagnosis_path)
    config = load_calibrated_config(config_path)
    scan_paths = [Path("src"), Path("scripts"), Path("tests")]
    old_repo_hits = []
    path_hacks = []
    for root in scan_paths:
        for path in root.rglob("*.py"):
            lines = path.read_text(encoding="utf-8").splitlines()
            for line in lines:
                stripped = line.strip()
                if stripped.startswith(("import ", "from ")) and any(
                    name in stripped
                    for name in OLD_REPO_NAMES
                ):
                    old_repo_hits.append(str(path))
                    break
            for line in lines:
                stripped = line.strip()
                if (stripped.startswith("sys.path") or stripped.startswith("os.environ[\"" + "PYTHON" + "PATH\"")) and "append" in stripped:
                    path_hacks.append(str(path))
                    break
    verdict = "READY"
    reasons = []
    if diagnosis.get("diagnosis") != "JUDGE_PROBLEM":
        verdict = "NOT_READY"
        reasons.append("failure diagnosis is not JUDGE_PROBLEM")
    if diagnosis.get("recommended_next_action") != "REPLACE_JUDGE":
        verdict = "NOT_READY"
        reasons.append("recommended next action is not REPLACE_JUDGE")
    if diagnosis.get("expansion_forbidden") is not True:
        verdict = "NOT_READY"
        reasons.append("expansion is not blocked")
    if old_repo_hits or path_hacks:
        verdict = "NOT_READY"
        reasons.append("forbidden dependency/path scan failed")
    result = {
        "v0_diagnosis": diagnosis.get("diagnosis"),
        "recommended_next_action": diagnosis.get("recommended_next_action"),
        "expansion_forbidden": bool(diagnosis.get("expansion_forbidden")),
        "calibrated_system_id": config["system_id"],
        "old_repo_runtime_import_hits": old_repo_hits,
        "path_hack_hits": path_hacks,
        "excluded_from_evidence": ["cstr", "heat_exchanger", "rssm"],
        "verdict": verdict,
        "reasons": reasons,
    }
    _write_json(out_dir / "precondition_check.json", result)
    text = f"""# Calibrated Judge Preconditions

## v0 diagnosis

{result["v0_diagnosis"]}; recommended next action: {result["recommended_next_action"]}.

## Expansion status

Expansion forbidden: {result["expansion_forbidden"]}.

## Forbidden dependency scan

Old repo hits: {old_repo_hits if old_repo_hits else "none"}
Path hack hits: {path_hacks if path_hacks else "none"}

## Existing systems excluded from evidence

{", ".join(result["excluded_from_evidence"])}

## Verdict

{verdict}
"""
    _ensure_dir(report_path.parent)
    report_path.write_text(text, encoding="utf-8")
    if verdict != "READY":
        raise RuntimeError(f"calibrated judge preconditions not ready: {reasons}")
    return result


def run_calibrated_seed_sweep(config_path: str | Path, seeds: list[int], output: str | Path) -> dict[str, Any]:
    base_config = load_calibrated_config(config_path)
    out_dir = Path(output)
    _ensure_dir(out_dir)
    all_risk = []
    seed_rows = []
    failures = []
    for seed in seeds:
        seed_dir = out_dir / f"seed_{seed}"
        config = dict(base_config)
        config["seed"] = int(seed)
        resolved = seed_dir / "resolved_config.yaml"
        _ensure_dir(seed_dir)
        resolved.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        try:
            summary = run_calibrated_judge(
                resolved,
                seed_dir,
                report_path=seed_dir / "calibrated_refusal_judge.md",
                command=f"python scripts/run_calibrated_seed_sweep.py --config {config_path} --seeds {' '.join(map(str, seeds))} --output {output}",
            )
            risk = pd.read_csv(seed_dir / "calibrated_risk_coverage.csv")
            risk.insert(0, "seed", seed)
            all_risk.append(risk)
            seed_rows.append(
                {
                    "seed": seed,
                    "verdict": summary["verdict"],
                    "best_calibrated_judge": summary["best_calibrated_judge"],
                    "low_coverage_win": bool(summary["low_coverage_win"]),
                    "leakage_detected": bool(summary["leakage_detected"]),
                }
            )
        except Exception as exc:  # pragma: no cover - integration failure path.
            failures.append({"seed": seed, "error": str(exc)})
            seed_rows.append(
                {
                    "seed": seed,
                    "verdict": "FAILED",
                    "best_calibrated_judge": "",
                    "low_coverage_win": False,
                    "leakage_detected": True,
                }
            )
    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(out_dir / "seed_sweep_calibrated_summary.csv", index=False)
    if failures:
        _write_json(out_dir / "failures.json", {"failures": failures})
    risk_all = pd.concat(all_risk, ignore_index=True) if all_risk else pd.DataFrame()
    risk_all.to_csv(out_dir / "calibrated_risk_coverage_all.csv", index=False)
    leakage = bool(seed_df["leakage_detected"].any()) if not seed_df.empty else True
    win_count = int(seed_df["low_coverage_win"].sum()) if not seed_df.empty else 0
    n = len(seed_df)
    if leakage:
        verdict = "INVALID_DUE_TO_LEAKAGE"
    elif win_count >= math.ceil(0.7 * n):
        verdict = "ROBUST_LOW_COVERAGE"
    elif win_count >= math.ceil(0.4 * n):
        verdict = "UNSTABLE"
    else:
        verdict = "NO_ROBUST_IMPROVEMENT"
    low = risk_all[risk_all["coverage_requested"].isin([float(v) for v in base_config["primary_coverages"]])] if not risk_all.empty else pd.DataFrame()
    baseline = low[low["judge_id"] == "best_single_signal_selected_on_calibration"]
    calibrated = low[low["judge_id"].isin([j for j in CALIBRATED_JUDGE_IDS if j != "best_single_signal_selected_on_calibration"])]
    agg_rows = []
    for coverage in [float(v) for v in base_config["primary_coverages"]]:
        b = baseline[baseline["coverage_requested"] == coverage]["false_accept_rate"].mean()
        c = calibrated[calibrated["coverage_requested"] == coverage].groupby(["seed", "model_id", "scenario_type"])["false_accept_rate"].min().mean()
        margins = []
        for seed in seeds:
            b_seed = baseline[(baseline["seed"] == seed) & (baseline["coverage_requested"] == coverage)]["false_accept_rate"].mean()
            c_seed = calibrated[(calibrated["seed"] == seed) & (calibrated["coverage_requested"] == coverage)].groupby(["model_id", "scenario_type"])["false_accept_rate"].min().mean()
            if not pd.isna(b_seed) and not pd.isna(c_seed):
                margins.append(float(b_seed - c_seed))
        agg_rows.append(
            {
                "coverage": coverage,
                "win_rate_vs_calibration_selected_single_signal": float(np.mean([m > 0 for m in margins])) if margins else 0.0,
                "mean_margin": float(np.mean(margins)) if margins else 0.0,
                "std_margin": float(np.std(margins)) if margins else 0.0,
            }
        )
    low_aggregate = pd.DataFrame(agg_rows)
    judge_robustness = (
        low.groupby("judge_id", as_index=False)
        .agg(win_count=("false_accept_rate", "count"), mean_far=("false_accept_rate", "mean"), std_far=("false_accept_rate", "std"))
        .fillna(0.0)
        if not low.empty
        else pd.DataFrame(columns=["judge_id", "win_count", "mean_far", "std_far"])
    )
    summary = {
        "command": f"python scripts/run_calibrated_seed_sweep.py --config {config_path} --seeds {' '.join(map(str, seeds))} --output {output}",
        "seeds": seeds,
        "verdict": verdict,
        "winning_seed_count": win_count,
        "leakage_detected": leakage,
        "failures": failures,
        "low_coverage_aggregate": low_aggregate.to_dict(orient="records"),
        "judge_robustness": judge_robustness.to_dict(orient="records"),
    }
    _write_json(out_dir / "seed_sweep_calibrated_summary.json", summary)
    write_calibrated_seed_sweep_report(summary, seed_df, low_aggregate, judge_robustness, Path("reports/calibrated_seed_sweep_report.md"))
    if failures:
        raise RuntimeError(f"calibrated seed sweep failed for {len(failures)} seeds")
    return summary


def write_calibrated_seed_sweep_report(summary: dict[str, Any], seed_df: pd.DataFrame, low: pd.DataFrame, judge: pd.DataFrame, output: Path) -> None:
    _ensure_dir(output.parent)
    failures = summary.get("failures", [])
    text = f"""# Calibrated Judge Seed Sweep Report

## Command

```bash
{summary["command"]}
```

## Seeds

{", ".join(map(str, summary["seeds"]))}

## Per-seed verdict

{_markdown_table(seed_df, ["seed", "verdict", "best_calibrated_judge", "low_coverage_win", "leakage_detected"])}

## Low-coverage aggregate

{_markdown_table(low, ["coverage", "win_rate_vs_calibration_selected_single_signal", "mean_margin", "std_margin"])}

## Judge robustness

{_markdown_table(judge.rename(columns={"judge_id": "judge"}), ["judge", "win_count", "mean_far", "std_far"])}

## Failure cases

{"none" if not failures else json.dumps(failures, indent=2)}

## Verdict

{summary["verdict"]}

## Explanation

Calibrated low-coverage wins appeared in {summary["winning_seed_count"]} of {len(summary["seeds"])} seeds.
"""
    output.write_text(text, encoding="utf-8")


def run_calibrated_stress(
    config_path: str | Path,
    thresholds: list[float],
    coverages: list[float],
    seeds: list[int],
    output: str | Path,
) -> dict[str, Any]:
    base_config = load_calibrated_config(config_path)
    out_dir = Path(output)
    _ensure_dir(out_dir)
    rows = []
    leakage = False
    for seed in seeds:
        seed_dir = out_dir / f"seed_{seed}_base"
        config = dict(base_config)
        config["seed"] = int(seed)
        config["coverages"] = [float(v) for v in coverages]
        resolved = seed_dir / "resolved_config.yaml"
        _ensure_dir(seed_dir)
        resolved.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        base_summary = run_calibrated_judge(resolved, seed_dir, report_path=None)
        leakage = leakage or bool(base_summary["leakage_detected"])
        calibration_table = pd.read_csv(seed_dir / "calibration_table.csv")
        test_table = pd.read_csv(seed_dir / "test_table.csv")
        for threshold in thresholds:
            stress_config = dict(config)
            stress_config["bad_threshold"] = dict(config["bad_threshold"])
            stress_config["bad_threshold"]["value"] = float(threshold)
            threshold_dir = out_dir / f"seed_{seed}_threshold_{threshold:g}"
            summary = evaluate_calibrated_tables(
                stress_config,
                calibration_table,
                test_table,
                threshold_dir,
                report_path=None,
                command="calibrated stress internal evaluation",
            )
            risk = pd.read_csv(threshold_dir / "calibrated_risk_coverage.csv")
            for coverage in coverages:
                group = risk[np.isclose(risk["coverage_requested"], float(coverage))]
                baseline_far = float(group[group["judge_id"] == "best_single_signal_selected_on_calibration"]["false_accept_rate"].mean())
                calibrated_far = float(
                    group[group["judge_id"].isin([j for j in CALIBRATED_JUDGE_IDS if j != "best_single_signal_selected_on_calibration"])]
                    .groupby(["model_id", "scenario_type"])["false_accept_rate"]
                    .min()
                    .mean()
                )
                bad_rates = risk[np.isclose(risk["coverage_requested"], float(coverage))].groupby(["model_id", "scenario_type"])["bad_count_total"].first()
                total_counts = risk[np.isclose(risk["coverage_requested"], float(coverage))].groupby(["model_id", "scenario_type"])["accepted_count"].max()
                degenerate = bool((bad_rates == 0).all())
                rows.append(
                    {
                        "seed": seed,
                        "threshold": float(threshold),
                        "coverage": float(coverage),
                        "baseline_far": baseline_far,
                        "best_calibrated_far": calibrated_far,
                        "margin": baseline_far - calibrated_far,
                        "win": bool(calibrated_far < baseline_far - 1e-12),
                        "degenerate": degenerate,
                        "single_run_verdict": summary["verdict"],
                        "leakage_detected": bool(summary["leakage_detected"]),
                    }
                )
                _ = total_counts
            leakage = leakage or bool(summary["leakage_detected"])
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "threshold_coverage_results.csv", index=False)
    valid = result[~result["degenerate"]]
    by_threshold = valid[valid["coverage"].isin([0.05, 0.10])].groupby("threshold", as_index=False)["win"].mean().rename(columns={"win": "low_coverage_win_rate"})
    by_threshold["verdict"] = np.where(by_threshold["low_coverage_win_rate"] >= 0.6, "WORKS", "FAILS")
    by_coverage = valid.groupby("coverage", as_index=False).agg(win_rate=("win", "mean"), mean_margin=("margin", "mean"))
    working_thresholds = by_threshold[by_threshold["low_coverage_win_rate"] >= 0.6]["threshold"].tolist()
    if leakage:
        verdict = "INVALID_DUE_TO_LEAKAGE"
    elif len(working_thresholds) >= 3:
        verdict = "ROBUST_LOW_COVERAGE_ONLY"
    elif 1 <= len(working_thresholds) <= 2:
        verdict = "THRESHOLD_DEPENDENT"
    else:
        verdict = "NO_STABLE_REGION"
    summary = {
        "command": f"python scripts/run_calibrated_stress.py --config {config_path} --thresholds {' '.join(map(str, thresholds))} --coverages {' '.join(map(str, coverages))} --seeds {' '.join(map(str, seeds))} --output {output}",
        "thresholds": thresholds,
        "coverages": coverages,
        "seeds": seeds,
        "verdict": verdict,
        "leakage_detected": leakage,
        "working_thresholds": working_thresholds,
        "result_by_threshold": by_threshold.to_dict(orient="records"),
        "result_by_coverage": by_coverage.to_dict(orient="records"),
    }
    _write_json(out_dir / "stress_summary.json", summary)
    write_calibrated_stress_report(summary, by_threshold, by_coverage, Path("reports/calibrated_threshold_coverage_stress.md"))
    return summary


def write_calibrated_stress_report(summary: dict[str, Any], by_threshold: pd.DataFrame, by_coverage: pd.DataFrame, output: Path) -> None:
    _ensure_dir(output.parent)
    works = by_threshold[by_threshold["low_coverage_win_rate"] >= 0.6]["threshold"].tolist() if not by_threshold.empty else []
    fails = by_threshold[by_threshold["low_coverage_win_rate"] < 0.6]["threshold"].tolist() if not by_threshold.empty else []
    text = f"""# Calibrated Judge Threshold/Coverage Stress Report

## Thresholds tested

{", ".join(map(str, summary["thresholds"]))}

## Coverages tested

{", ".join(map(str, summary["coverages"]))}

## Seeds tested

{", ".join(map(str, summary["seeds"]))}

## Result by threshold

{_markdown_table(by_threshold, ["threshold", "low_coverage_win_rate", "verdict"])}

## Result by coverage

{_markdown_table(by_coverage, ["coverage", "win_rate", "mean_margin"])}

## Regions where calibrated judge works

{", ".join(map(str, works)) if works else "none"}

## Regions where calibrated judge fails

{", ".join(map(str, fails)) if fails else "none"}

## Verdict

{summary["verdict"]}
"""
    output.write_text(text, encoding="utf-8")


def make_calibrated_decision_gate(single_run: str | Path, seed_sweep: str | Path, stress: str | Path, output: str | Path) -> dict[str, Any]:
    single = _load_json(Path(single_run))
    seeds = _load_json(Path(seed_sweep))
    stress_data = _load_json(Path(stress))
    leakage = bool(single.get("leakage_detected") or seeds.get("leakage_detected") or stress_data.get("leakage_detected"))
    if leakage:
        decision = "INVALID_DUE_TO_LEAKAGE"
    elif (
        single["verdict"] == "SUPPORTED_LOW_COVERAGE"
        and seeds["verdict"] == "ROBUST_LOW_COVERAGE"
        and stress_data["verdict"] == "ROBUST_LOW_COVERAGE_ONLY"
    ):
        decision = "PROCEED_TO_CSTR"
    elif (
        single["verdict"] in {"SUPPORTED_LOW_COVERAGE", "MIXED"}
        and seeds["verdict"] in {"ROBUST_LOW_COVERAGE", "UNSTABLE"}
        and stress_data["verdict"] in {"ROBUST_LOW_COVERAGE_ONLY", "THRESHOLD_DEPENDENT"}
    ):
        decision = "KEEP_WITH_LOW_COVERAGE_CLAIM"
    elif single["verdict"] == "NO_IMPROVEMENT_OVER_SINGLE_SIGNAL" or seeds["verdict"] == "NO_ROBUST_IMPROVEMENT":
        decision = "NO_IMPROVEMENT_OVER_SINGLE_SIGNAL"
    else:
        decision = "KILL_OR_DOWNGRADE_FURTHER"
    allowed = ["replace calibrated judge and rerun TwoTank evidence"]
    forbidden = ["CSTR evidence", "RSSM", "new systems", "platform/product work"]
    if decision == "PROCEED_TO_CSTR":
        allowed = ["run a separately gated CSTR replication"]
        forbidden = ["RSSM", "platform/product work"]
    result = {
        "single_run_verdict": single["verdict"],
        "seed_sweep_verdict": seeds["verdict"],
        "stress_verdict": stress_data["verdict"],
        "leakage_detected": leakage,
        "decision": decision,
        "allowed_next_actions": allowed,
        "forbidden_next_actions": forbidden,
    }
    output_path = Path(output)
    _ensure_dir(output_path.parent)
    _write_json(output_path.with_suffix(".json"), result)
    text = f"""# Calibrated Judge Decision Gate

## Starting point

The v0 combined_linear claim was not supported. This gate tests whether calibrated refusal replaces it.

## Single-run result

{single["verdict"]}

## Seed-sweep result

{seeds["verdict"]}

## Threshold/coverage stress result

{stress_data["verdict"]}

## Leakage status

{leakage}

## Decision

{decision}

## Allowed next actions

{", ".join(allowed)}

## Forbidden next actions

{", ".join(forbidden)}

## Explanation

Decision follows the calibrated-judge gate rules. Oracle is diagnostic only and is not used as evidence.
"""
    output_path.write_text(text, encoding="utf-8")
    return result
