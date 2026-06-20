from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from scs.data.generate import generate_and_save_dataset, summarize_dataset
from scs.data.splits import action_range, assert_not_identical, max_inflow
from scs.experiments.registry import make_model, make_system
from scs.metrics.selective import risk_coverage_curve
from scs.metrics.trajectory import final_state_error, mae, max_abs_error, rmse
from scs.reports.plots import plot_risk_coverage
from scs.reports.summary import write_smoke_report, write_summary_json
from scs.validators.disagreement import disagreement_score
from scs.validators.invariants import invariant_residual_score
from scs.validators.judges import JUDGE_IDS, compute_judge_score_frame
from scs.validators.repair import repair_amount_score
from scs.validators.support import SupportDistance
from scs.validators.uncertainty import uncertainty_score


REQUIRED_CONFIG_KEYS = {
    "experiment_id",
    "seed",
    "horizon",
    "dt",
    "n_train",
    "n_id_test",
    "n_ood_test",
    "models",
    "judges",
    "bad_threshold",
    "coverages",
}


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("config must be a mapping")
    required = set(REQUIRED_CONFIG_KEYS)
    if "systems" not in config:
        required.add("system_id")
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"missing required config keys: {missing}")
    if "systems" in config and not config["systems"]:
        raise ValueError("systems must contain at least one system")
    return config


def _output_dir_for(config: dict[str, Any], output_dir: str | Path | None = None) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    return Path(str(config.get("output_dir", f"results/{config['experiment_id']}")))


def _validate_shift(dataset: dict) -> None:
    train = dataset["train"]
    id_test = dataset["id_test"]
    assert_not_identical(train, id_test)
    action_train = action_range(train)
    action_ood = action_range(dataset["ood_action_magnitude"])
    if action_ood <= action_train * 1.25:
        raise RuntimeError(
            f"OOD action range is not sufficiently different: train={action_train:.4f}, ood={action_ood:.4f}"
        )
    inflow_train = max_inflow(train)
    inflow_ood = max_inflow(dataset["ood_inflow_spike"])
    if inflow_ood <= inflow_train * 1.25:
        raise RuntimeError(
            f"OOD inflow max is not sufficiently different: train={inflow_train:.4f}, ood={inflow_ood:.4f}"
        )


def _validate_predictions_not_identical(prediction_vectors: dict[str, np.ndarray]) -> None:
    model_ids = list(prediction_vectors)
    any_different = False
    for left, right in combinations(model_ids, 2):
        if not np.allclose(prediction_vectors[left], prediction_vectors[right]):
            any_different = True
            break
    if not any_different:
        raise RuntimeError("all models produced identical predictions")


def _validate_judges_not_identical(scenario_scores: pd.DataFrame, judge_ids: list[str]) -> None:
    ranking_columns = [f"risk_{judge_id}" for judge_id in judge_ids]
    for model_id, model_df in scenario_scores.groupby("model_id"):
        rankings = {
            tuple(np.argsort(model_df[column].to_numpy(dtype=float), kind="mergesort"))
            for column in ranking_columns
        }
        if len(rankings) <= 1:
            raise RuntimeError(f"all judges produced identical ranking for {model_id}")


def _combined_judge_statement(risk_coverage: pd.DataFrame) -> dict[str, Any]:
    simple = [
        "support_only",
        "uncertainty_only",
        "disagreement_only",
        "invariant_only",
        "repair_only",
    ]
    aggregate = (
        risk_coverage.groupby(["judge_id", "coverage"], as_index=False)["false_accept_rate"]
        .mean()
    )
    rows = []
    beats_all = True
    for coverage in sorted(aggregate["coverage"].unique()):
        combined = float(
            aggregate[
                (aggregate["judge_id"] == "combined_linear")
                & (np.isclose(aggregate["coverage"], coverage))
            ]["false_accept_rate"].iloc[0]
        )
        simple_best = float(
            aggregate[
                (aggregate["judge_id"].isin(simple))
                & (np.isclose(aggregate["coverage"], coverage))
            ]["false_accept_rate"].min()
        )
        beat = combined <= simple_best + 1e-12
        beats_all = beats_all and beat
        rows.append(
            {
                "coverage": float(coverage),
                "combined_false_accept_rate": combined,
                "best_simple_false_accept_rate": simple_best,
                "combined_beat_best_simple": bool(beat),
            }
        )
    if beats_all:
        statement = "Combined_linear matched or beat the strongest single-signal judge at every configured coverage in this smoke run."
    else:
        statement = "Combined_linear did not beat the strongest single-signal judge at every configured coverage in this smoke run."
    return {"statement": statement, "by_coverage": rows}


def _claim_status(risk_coverage: pd.DataFrame) -> dict[str, Any]:
    per_system = {}
    supported_count = 0
    for system_id, system_df in risk_coverage.groupby("system_id"):
        result = _combined_judge_statement(system_df)
        supported = all(row["combined_beat_best_simple"] for row in result["by_coverage"])
        supported_count += int(supported)
        per_system[system_id] = {
            "supported": bool(supported),
            "statement": result["statement"],
            "by_coverage": result["by_coverage"],
        }

    total = len(per_system)
    if supported_count == total and total > 0:
        status = "SUPPORTED"
    elif supported_count == 0:
        status = "NOT SUPPORTED"
    else:
        status = "MIXED"
    return {
        "result": status,
        "explanation": (
            f"Combined_linear beat or matched the strongest simple judge on "
            f"{supported_count} of {total} evaluated systems."
        ),
        "per_system": per_system,
    }


def _build_model_metrics(scenario_scores: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        scenario_scores.groupby(["system_id", "model_id", "split"], as_index=False)
        .agg(
            rmse_mean=("error", "mean"),
            rmse_median=("error", "median"),
            mae_mean=("mae", "mean"),
            max_abs_error_mean=("max_abs_error", "mean"),
            final_state_error_mean=("final_state_error", "mean"),
            n_scenarios=("scenario_id", "count"),
        )
        .sort_values(["model_id", "split"])
    )
    return metrics


def _threshold_for_system(config: dict[str, Any], system_id: str) -> dict[str, Any]:
    threshold = dict(config["bad_threshold"])
    per_system = threshold.get("per_system")
    if isinstance(per_system, dict) and system_id in per_system:
        threshold["value"] = per_system[system_id]
    threshold.pop("per_system", None)
    return threshold


def _resolved_system_config(base_config: dict[str, Any], system_entry: str | dict[str, Any], out_dir: Path) -> dict[str, Any]:
    if isinstance(system_entry, str):
        system_id = system_entry
        overrides: dict[str, Any] = {}
    elif isinstance(system_entry, dict):
        system_id = str(system_entry["system_id"])
        overrides = {key: value for key, value in system_entry.items() if key != "system_id"}
    else:
        raise ValueError("each systems entry must be a system id string or mapping")

    single = {
        key: value
        for key, value in base_config.items()
        if key not in {"systems", "system_id", "output_dir", "legacy_data_dir"}
    }
    single.update(overrides)
    single["system_id"] = system_id
    single["experiment_id"] = f"{base_config['experiment_id']}_{system_id}"
    single["output_dir"] = str(out_dir / system_id)
    single["bad_threshold"] = _threshold_for_system(base_config, system_id)
    return single


def _run_multi_system_experiment(
    config: dict[str, Any],
    output_dir: str | Path | None,
    report_path: str | Path,
    command: str,
) -> dict[str, Any]:
    seed = int(config["seed"])
    out_dir = _output_dir_for(config, output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_system_summaries: dict[str, dict[str, Any]] = {}
    risk_frames = []
    scenario_frames = []
    metric_frames = []
    per_system_plots: dict[str, str] = {}

    for system_index, system_entry in enumerate(config["systems"]):
        single_config = _resolved_system_config(config, system_entry, out_dir)
        single_config["seed"] = seed + system_index * 10000
        system_id = str(single_config["system_id"])
        resolved_config_path = out_dir / f"{system_id}_resolved_config.yaml"
        with resolved_config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(single_config, handle, sort_keys=False)

        system_report_path = out_dir / f"{system_id}_report.md"
        system_summary = run_experiment(
            resolved_config_path,
            output_dir=out_dir / system_id,
            report_path=system_report_path,
            command=f"{command} --system {system_id}",
        )
        per_system_summaries[system_id] = system_summary

        system_risk = pd.read_csv(out_dir / system_id / "risk_coverage.csv")
        system_scores = pd.read_csv(out_dir / system_id / "scenario_scores.csv")
        system_metrics = pd.read_csv(out_dir / system_id / "model_metrics.csv")
        risk_frames.append(system_risk)
        scenario_frames.append(system_scores)
        metric_frames.append(system_metrics)

        per_system_plot = out_dir / f"risk_coverage_{system_id}.png"
        plot_risk_coverage(system_risk, per_system_plot)
        per_system_plots[system_id] = str(per_system_plot)

    risk_coverage = pd.concat(risk_frames, ignore_index=True)
    scenario_scores = pd.concat(scenario_frames, ignore_index=True)
    model_metrics = pd.concat(metric_frames, ignore_index=True)

    if not np.isfinite(risk_coverage.select_dtypes(include=[float, int]).to_numpy()).all():
        raise RuntimeError("multi-system risk_coverage contains NaN or infinite values")
    if not np.isfinite(scenario_scores.select_dtypes(include=[float, int]).to_numpy()).all():
        raise RuntimeError("multi-system scenario_scores contains NaN or infinite values")

    risk_coverage.to_csv(out_dir / "risk_coverage.csv", index=False)
    scenario_scores.to_csv(out_dir / "scenario_scores.csv", index=False)
    model_metrics.to_csv(out_dir / "model_metrics.csv", index=False)
    plot_risk_coverage(risk_coverage, out_dir / "risk_coverage.png")

    data_summary = {
        system_id: summary["dataset_summary"]
        for system_id, summary in per_system_summaries.items()
    }
    with (out_dir / "data_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(data_summary, handle, indent=2, sort_keys=True)

    summary = {
        "config": config,
        "systems": sorted(per_system_summaries),
        "dataset_summary": data_summary,
        "per_system": per_system_summaries,
        "model_metrics": model_metrics.to_dict(orient="records"),
        "risk_coverage_rows": int(len(risk_coverage)),
        "scenario_score_rows": int(len(scenario_scores)),
        "combined_judge_result": _combined_judge_statement(risk_coverage),
        "claim_status": _claim_status(risk_coverage),
        "known_failures": [],
        "artifacts": {
            "data_summary": str(out_dir / "data_summary.json"),
            "model_metrics": str(out_dir / "model_metrics.csv"),
            "scenario_scores": str(out_dir / "scenario_scores.csv"),
            "risk_coverage": str(out_dir / "risk_coverage.csv"),
            "risk_coverage_plot": str(out_dir / "risk_coverage.png"),
            "per_system_risk_coverage_plots": per_system_plots,
            "summary": str(out_dir / "summary.json"),
            "smoke_report": str(report_path),
        },
    }
    write_summary_json(summary, out_dir / "summary.json")
    write_smoke_report(summary, risk_coverage, model_metrics, report_path, command=command)

    required_artifacts = [
        out_dir / "data_summary.json",
        out_dir / "model_metrics.csv",
        out_dir / "scenario_scores.csv",
        out_dir / "risk_coverage.csv",
        out_dir / "risk_coverage.png",
        out_dir / "summary.json",
        Path(report_path),
    ]
    for plot_path in per_system_plots.values():
        required_artifacts.append(Path(plot_path))
    missing = [str(path) for path in required_artifacts if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"missing required multi-system artifacts: {missing}")
    return summary


def run_experiment(
    config_path: str | Path,
    output_dir: str | Path | None = None,
    report_path: str | Path = "reports/smoke_report.md",
    command: str = "python scripts/run_smoke.py",
) -> dict[str, Any]:
    config = load_config(config_path)
    if report_path == "reports/smoke_report.md" and config["experiment_id"] != "smoke_two_tank":
        report_path = f"reports/{config['experiment_id']}_report.md"
    if "systems" in config:
        return _run_multi_system_experiment(config, output_dir, report_path, command)

    seed = int(config["seed"])
    np.random.seed(seed)
    out_dir = _output_dir_for(config, output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = generate_and_save_dataset(config, out_dir)
    _validate_shift(dataset)

    system = make_system(str(config["system_id"]))
    support = SupportDistance()
    support.fit(dataset["train"])

    models = [make_model(model_id, seed=seed + idx) for idx, model_id in enumerate(config["models"])]
    for model in models:
        model.fit(dataset["train"])

    test_split_names = [name for name in dataset if name != "train"]
    predictions: dict[str, dict[tuple[str, int], np.ndarray]] = {model.model_id: {} for model in models}
    true_by_key: dict[tuple[str, int], np.ndarray] = {}
    actions_by_key: dict[tuple[str, int], np.ndarray] = {}
    disturbances_by_key: dict[tuple[str, int], np.ndarray] = {}
    scenario_type_by_key: dict[tuple[str, int], str] = {}

    for split in test_split_names:
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

    prediction_vectors = {
        model_id: np.concatenate([predictions[model_id][key].ravel() for key in sorted(true_by_key)])
        for model_id in predictions
    }
    _validate_predictions_not_identical(prediction_vectors)

    rows: list[dict[str, Any]] = []
    n_samples = int(config.get("uncertainty_samples", 6))
    for key in sorted(true_by_key):
        split, i = key
        all_predictions = [predictions[model.model_id][key] for model in models]
        scenario_disagreement = disagreement_score(all_predictions)
        for model in models:
            predicted = predictions[model.model_id][key]
            actual = true_by_key[key]
            row = {
                "scenario_id": f"{split}_{i:04d}",
                "system_id": system.system_id,
                "split": split,
                "scenario_type": scenario_type_by_key[key],
                "model_id": model.model_id,
                "error": rmse(predicted, actual),
                "mae": mae(predicted, actual),
                "max_abs_error": max_abs_error(predicted, actual),
                "final_state_error": final_state_error(predicted, actual),
                "support_distance": support.score(actions_by_key[key], disturbances_by_key[key]),
                "uncertainty": uncertainty_score(
                    model,
                    actual[0],
                    actions_by_key[key],
                    disturbances_by_key[key],
                    n_samples=n_samples,
                ),
                "disagreement": scenario_disagreement,
                "invariant_residual": invariant_residual_score(
                    system,
                    predicted,
                    actions_by_key[key],
                    disturbances_by_key[key],
                    float(config["dt"]),
                ),
                "repair_amount": repair_amount_score(system, predicted),
            }
            rows.append(row)

    scenario_scores = pd.DataFrame(rows)
    if scenario_scores.empty:
        raise RuntimeError("scenario_scores is empty")
    if not np.isfinite(scenario_scores.select_dtypes(include=[float, int]).to_numpy()).all():
        raise RuntimeError("scenario_scores contains NaN or infinite values")

    judge_ids = [str(judge_id) for judge_id in config["judges"]]
    if len(judge_ids) < 6:
        raise RuntimeError("at least six judges are required")
    unknown_judges = sorted(set(judge_ids) - set(JUDGE_IDS))
    if unknown_judges:
        raise ValueError(f"unknown judges: {unknown_judges}")

    scored_groups = []
    for model_idx, (model_id, model_df) in enumerate(scenario_scores.groupby("model_id", sort=False)):
        judge_scores = compute_judge_score_frame(
            model_df.reset_index(drop=True),
            judge_ids=judge_ids,
            seed=seed + 5000 + model_idx,
        )
        enriched = model_df.reset_index(drop=True).copy()
        for judge_id in judge_ids:
            enriched[f"risk_{judge_id}"] = judge_scores[judge_id].to_numpy(dtype=float)
        scored_groups.append(enriched)
    scenario_scores = pd.concat(scored_groups, ignore_index=True)
    _validate_judges_not_identical(scenario_scores, judge_ids)

    bad_threshold = float(config["bad_threshold"]["value"])
    coverages = [float(value) for value in config["coverages"]]
    risk_rows = []
    for model_id, model_df in scenario_scores.groupby("model_id", sort=False):
        for judge_id in judge_ids:
            curve = risk_coverage_curve(
                errors=model_df["error"].to_numpy(dtype=float),
                risk_scores=model_df[f"risk_{judge_id}"].to_numpy(dtype=float),
                bad_threshold=bad_threshold,
                coverages=coverages,
            )
            curve.insert(0, "judge_id", judge_id)
            curve.insert(0, "model_id", model_id)
            curve.insert(0, "system_id", system.system_id)
            risk_rows.append(curve)
    risk_coverage = pd.concat(risk_rows, ignore_index=True)
    if risk_coverage.empty:
        raise RuntimeError("risk_coverage is empty")
    if not np.isfinite(risk_coverage.select_dtypes(include=[float, int]).to_numpy()).all():
        raise RuntimeError("risk_coverage contains NaN or infinite values")

    model_metrics = _build_model_metrics(scenario_scores)
    id_metrics = model_metrics[model_metrics["split"] == "id_test"].set_index("model_id")
    if {"hold_last", "linear_narx"}.issubset(id_metrics.index):
        if float(id_metrics.loc["linear_narx", "rmse_mean"]) >= float(id_metrics.loc["hold_last", "rmse_mean"]):
            raise RuntimeError("LinearNARX did not beat HoldLast on ID TwoTank RMSE")

    scenario_scores.to_csv(out_dir / "scenario_scores.csv", index=False)
    risk_coverage.to_csv(out_dir / "risk_coverage.csv", index=False)
    model_metrics.to_csv(out_dir / "model_metrics.csv", index=False)
    plot_risk_coverage(risk_coverage, out_dir / "risk_coverage.png")

    summary = {
        "config": config,
        "dataset_summary": summarize_dataset(dataset),
        "model_metrics": model_metrics.to_dict(orient="records"),
        "risk_coverage_rows": int(len(risk_coverage)),
        "scenario_score_rows": int(len(scenario_scores)),
        "combined_judge_result": _combined_judge_statement(risk_coverage),
        "known_failures": [],
        "artifacts": {
            "data_summary": str(out_dir / "data_summary.json"),
            "model_metrics": str(out_dir / "model_metrics.csv"),
            "scenario_scores": str(out_dir / "scenario_scores.csv"),
            "risk_coverage": str(out_dir / "risk_coverage.csv"),
            "risk_coverage_plot": str(out_dir / "risk_coverage.png"),
            "summary": str(out_dir / "summary.json"),
            "smoke_report": str(report_path),
        },
    }
    write_summary_json(summary, out_dir / "summary.json")
    write_smoke_report(summary, risk_coverage, model_metrics, report_path, command=command)

    required_artifacts = [
        out_dir / "data_summary.json",
        out_dir / "model_metrics.csv",
        out_dir / "scenario_scores.csv",
        out_dir / "risk_coverage.csv",
        out_dir / "risk_coverage.png",
        out_dir / "summary.json",
        Path(report_path),
    ]
    missing = [str(path) for path in required_artifacts if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"missing required artifacts: {missing}")

    return summary


def load_results(results_dir: str | Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    source = Path(results_dir)
    with (source / "summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    risk_coverage = pd.read_csv(source / "risk_coverage.csv")
    model_metrics = pd.read_csv(source / "model_metrics.csv")
    return summary, risk_coverage, model_metrics
