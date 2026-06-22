from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scs.data.generate import (
    generate_calibrated_cstr_dataset,
    generate_calibrated_heat_exchanger_dataset,
    generate_calibrated_two_tank_dataset,
    summarize_dataset,
)
from scs.experiments.registry import make_system
from scs.metrics.trajectory import final_state_error, mae, max_abs_error, rmse
from scs.models.ensemble_mlp import EnsembleMLPModel
from scs.models.gradient_boosted_narx import GradientBoostedNARXModel
from scs.models.hold_last import HoldLastModel
from scs.models.linear_narx import LinearNARXModel
from scs.models.mlp_state_space import MLPStateSpaceModel
from scs.systems.base import TrajectoryBatch
from scs.validators.disagreement import disagreement_score
from scs.validators.invariants import invariant_residual_score
from scs.validators.repair import repair_amount_score
from scs.validators.support import SupportDistance
from scs.validators.uncertainty import uncertainty_score


V2_ID = "v2_scientific_strengthening"
V2_ROOT = Path("results/v2_scientific_strengthening")
V2_ALLOWED_V1_CLAIM = "A weak but positive low-coverage result under the frozen protocol."
V2_REQUIRED_DECISIONS = {
    "UPGRADE_TO_MODERATE_MULTI_SYSTEM_LOW_COVERAGE_CLAIM",
    "KEEP_WEAK_LOW_COVERAGE_BENCHMARK_CLAIM",
    "SYSTEM_DEPENDENT_BENCHMARK_RESULT",
    "NO_METHOD_CLAIM_BENCHMARK_ONLY",
    "INVALID_V2_PROTOCOL",
}
V2_STAT_VERDICTS = {
    "STRONG_MULTI_SYSTEM_EFFECT",
    "WEAK_MULTI_SYSTEM_EFFECT",
    "MIXED_SYSTEM_DEPENDENT_EFFECT",
    "NO_ROBUST_EFFECT",
    "INVALID_DUE_TO_LEAKAGE_OR_BENCHMARK_FAILURE",
}
SIGNAL_COLUMNS = [
    "support_distance",
    "uncertainty_score",
    "disagreement_score",
    "invariant_residual",
    "repair_amount",
]
SIMPLE_RISK_COLUMN = {
    "support_only": "support_distance",
    "uncertainty_only": "uncertainty_score",
    "disagreement_only": "disagreement_score",
    "invariant_only": "invariant_residual",
    "repair_only": "repair_amount",
}
PRODUCT_DIRS = {"frontend", "dashboard", "api", "app", "database", "auth"}
OLD_REPO_NAMES = [
    "time" + "-series" + "-simulator",
    "digital" + "-twin" + "-engine",
    "flux" + "-attention" + "-engine",
    "plant" + "-scenario" + "-compiler",
]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    _ensure_dir(path.parent)
    frame.to_csv(path, index=False)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data


def load_v2_config(path: str | Path) -> dict[str, Any]:
    config = _load_yaml(path)
    required = {
        "v2_id",
        "source_v1_tag",
        "systems",
        "models",
        "signals",
        "judges",
        "primary_coverages",
        "coverage_grid",
        "badness_targets",
        "rmse_thresholds",
        "seeds",
        "practical_thresholds",
        "forbidden",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"missing v2 config keys: {missing}")
    if config["v2_id"] != V2_ID:
        raise ValueError(f"unexpected v2_id: {config['v2_id']}")
    forbidden = config.get("forbidden", {})
    if forbidden.get("allow_product_features") is not False:
        raise ValueError("v2 config must forbid product features")
    if forbidden.get("allow_oracle_as_real_method") is not False:
        raise ValueError("v2 config must forbid oracle as a real method")
    if forbidden.get("allow_protocol_mutation_after_results") is not False:
        raise ValueError("v2 config must forbid protocol mutation after results")
    if forbidden.get("allow_v1_artifact_overwrite") is not False:
        raise ValueError("v2 config must forbid v1 artifact overwrite")
    if len(config["seeds"]) < 10 and not bool(config.get("test_mode", False)):
        raise ValueError("v2 config must contain at least 10 frozen seeds")
    if len(config["coverage_grid"]) < 5 or not set(config["primary_coverages"]).issubset(config["coverage_grid"]):
        raise ValueError("v2 config coverage grid is incomplete")
    if len(config["rmse_thresholds"]) < 6:
        raise ValueError("v2 config threshold grid is incomplete")
    return config


def load_event_config(path: str | Path) -> dict[str, Any]:
    config = _load_yaml(path)
    for system_id, system_config in config.items():
        thresholds = system_config.get("thresholds", {})
        for key, value in thresholds.items():
            if isinstance(value, str):
                raise ValueError(f"placeholder event threshold for {system_id}.{key}: {value}")
            float(value)
    return config


def _git_output(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return "unknown"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    table = frame[columns].copy()
    if max_rows is not None:
        table = table.head(max_rows)
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


def _scan_forbidden_runtime_imports() -> dict[str, list[str]]:
    old_repo_hits: list[str] = []
    path_hack_hits: list[str] = []
    for root in [Path("src"), Path("scripts"), Path("tests")]:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith(("import ", "from ")) and any(name in stripped for name in OLD_REPO_NAMES):
                    old_repo_hits.append(str(path))
                if stripped.startswith("sys.path"):
                    path_hack_hits.append(str(path))
    return {
        "old_repo_runtime_import_hits": sorted(set(old_repo_hits)),
        "path_hack_hits": sorted(set(path_hack_hits)),
    }


def _protocol_hash(protocol_path: Path = Path("docs/v2/v2_scientific_protocol_lock.md")) -> str:
    return _sha256_file(protocol_path) if protocol_path.exists() else ""


def verify_v2_preconditions(config_path: str | Path, output: str | Path) -> dict[str, Any]:
    config = load_v2_config(config_path)
    out_dir = Path(output)
    _ensure_dir(out_dir)

    source_artifacts = [Path(path) for path in config.get("source_artifacts", [])]
    source_hashes = {
        str(path): _sha256_file(path)
        for path in source_artifacts
        if path.exists()
    }
    missing_source_artifacts = [str(path) for path in source_artifacts if not path.exists()]
    v1_tag = str(config["source_v1_tag"])
    tag_exists = v1_tag in _git_output(["tag", "--list", v1_tag]).splitlines()
    forbidden_scan = _scan_forbidden_runtime_imports()
    product_dirs = sorted(str(path) for path in Path(".").iterdir() if path.is_dir() and path.name in PRODUCT_DIRS)
    v2_separate = str(out_dir).startswith(str(V2_ROOT))
    heat_exists = Path("src/scs/systems/heat_exchanger.py").exists()
    status_before_write = _git_output(["status", "--short"])
    reasons: list[str] = []
    if not tag_exists:
        reasons.append("source v1 tag missing")
    if missing_source_artifacts:
        reasons.append("source v1 artifacts missing")
    if not v2_separate:
        reasons.append("v2 output directory is not separate")
    if forbidden_scan["old_repo_runtime_import_hits"] or forbidden_scan["path_hack_hits"]:
        reasons.append("forbidden runtime dependency scan failed")
    if product_dirs:
        reasons.append("product directories detected")
    if "oracle_error_rank" not in config["judges"]["diagnostic_only"]:
        reasons.append("oracle is not listed as diagnostic-only")
    if not heat_exists:
        reasons.append("heat_exchanger implementation missing")

    verdict = "READY_FOR_V2_PROTOCOL_LOCK" if not reasons else "NOT_READY_FOR_V2_PROTOCOL_LOCK"
    result = {
        "v2_id": config["v2_id"],
        "source_v1_tag": v1_tag,
        "v1_tag_exists": tag_exists,
        "source_artifacts_checked": [str(path) for path in source_artifacts],
        "missing_source_artifacts": missing_source_artifacts,
        "v1_artifact_hash_count": len(source_hashes),
        "working_tree_status_before_write": status_before_write,
        "v2_output_directory": str(out_dir),
        "v2_output_directory_separate": v2_separate,
        "heat_exchanger_exists": heat_exists,
        "heat_exchanger_validated": False,
        "forbidden_dependency_scan": forbidden_scan,
        "product_directories": product_dirs,
        "oracle_diagnostic_only": "oracle_error_rank" in config["judges"]["diagnostic_only"],
        "protocol_hash_if_present": _protocol_hash(),
        "verdict": verdict,
        "reasons": reasons,
    }
    _write_json(out_dir / "v1_artifact_hashes.json", source_hashes)
    _write_json(out_dir / "precondition_check.json", result)
    report = f"""# v2 Precondition Check

## Verdict

{verdict}

## v1 tag

{v1_tag}: {tag_exists}

## v1 source artifacts

Recorded hashes: {len(source_hashes)}
Missing artifacts: {missing_source_artifacts}

## Working tree

```text
{status_before_write or "clean"}
```

## Separation and dependency checks

v2 output directory separate: {v2_separate}

Heat exchanger exists but is not yet validated: {heat_exists}

Oracle diagnostic only: {result["oracle_diagnostic_only"]}

Product directories: {product_dirs}

Forbidden dependency scan: {forbidden_scan}

## Reasons

{reasons or ["none"]}
"""
    Path("reports/v2_precondition_check.md").write_text(report, encoding="utf-8")
    return result


def generate_v2_dataset(system_id: str, config: dict[str, Any], seed: int) -> dict[str, TrajectoryBatch]:
    data = config["data"]
    kwargs = {
        "n_model_train": int(data["n_model_train"]),
        "n_calibration_id": int(data["n_calibration_id"]),
        "n_calibration_ood": int(data["n_calibration_ood"]),
        "n_test_id": int(data["n_test_id"]),
        "n_test_ood": int(data["n_test_ood"]),
        "horizon": int(data["horizon"]),
        "dt": float(data["dt"]),
        "seed": int(seed),
    }
    if system_id == "two_tank":
        return generate_calibrated_two_tank_dataset(**kwargs)
    if system_id == "cstr":
        return generate_calibrated_cstr_dataset(**kwargs)
    if system_id == "heat_exchanger":
        return generate_calibrated_heat_exchanger_dataset(**kwargs)
    raise ValueError(f"unknown v2 system_id: {system_id}")


def _make_v2_model(model_id: str, seed: int, config: dict[str, Any]):
    runtime = config.get("model_runtime", {})
    if model_id == "hold_last":
        return HoldLastModel()
    if model_id == "linear_narx":
        return LinearNARXModel(random_state=seed)
    if model_id == "mlp_state_space":
        return MLPStateSpaceModel(
            random_state=seed,
            hidden_layer_sizes=(32,),
            max_iter=int(runtime.get("mlp_max_iter", 80)),
        )
    if model_id == "ensemble_mlp":
        return EnsembleMLPModel(
            random_state=seed,
            n_members=int(runtime.get("ensemble_members", 2)),
            max_iter=int(runtime.get("ensemble_mlp_max_iter", 70)),
        )
    if model_id == "gradient_boosted_narx":
        return GradientBoostedNARXModel(
            random_state=seed,
            max_iter=int(runtime.get("gradient_boosting_max_iter", 60)),
        )
    raise ValueError(f"unknown v2 model_id: {model_id}")


def _split_role(split: str) -> str:
    if split == "model_train":
        return "model_train"
    if split.startswith("judge_calibration"):
        return "judge_calibration"
    if split.startswith("judge_test"):
        return "judge_test"
    raise ValueError(f"unknown v2 split role: {split}")


def _batch_hash(batch: TrajectoryBatch, index: int) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(batch.states[index]).tobytes())
    digest.update(np.ascontiguousarray(batch.actions[index]).tobytes())
    digest.update(np.ascontiguousarray(batch.disturbances[index]).tobytes())
    return digest.hexdigest()


def split_overlap_report(dataset: dict[str, TrajectoryBatch]) -> dict[str, Any]:
    calibration_hashes: set[str] = set()
    test_hashes: set[str] = set()
    for split, batch in dataset.items():
        hashes = {_batch_hash(batch, idx) for idx in range(batch.n_trajectories)}
        if _split_role(split) == "judge_calibration":
            calibration_hashes.update(hashes)
        elif _split_role(split) == "judge_test":
            test_hashes.update(hashes)
    overlap = sorted(calibration_hashes.intersection(test_hashes))
    return {
        "calibration_hash_count": len(calibration_hashes),
        "test_hash_count": len(test_hashes),
        "overlap_count": len(overlap),
        "overlap_examples": overlap[:5],
    }


def trajectory_event_flags(system_id: str, states: np.ndarray, event_config: dict[str, Any]) -> dict[str, bool]:
    thresholds = event_config[system_id]["thresholds"]
    states = np.asarray(states, dtype=float)
    if system_id == "two_tank":
        level_min = float(thresholds["level_min"])
        level_max = float(thresholds["level_max"])
        return {
            "overflow_event": bool(np.any(states > level_max)),
            "underflow_event": bool(np.any(states < level_min)),
        }
    if system_id == "cstr":
        temp_high = float(thresholds["temperature_high"])
        c_low = float(thresholds["concentration_low"])
        c_high = float(thresholds["concentration_high"])
        temperature_above = bool(np.any(states[:, 1] > temp_high))
        concentration_out = bool(np.any((states[:, 0] < c_low) | (states[:, 0] > c_high)))
        return {
            "temperature_above_limit": temperature_above,
            "concentration_out_of_safe_range": concentration_out,
            "unsafe_reactor_state": bool(temperature_above or concentration_out),
        }
    if system_id == "heat_exchanger":
        high = float(thresholds["outlet_temperature_high"])
        low = float(thresholds["outlet_temperature_low"])
        target = float(thresholds["target_outlet_temperature"])
        tracking = float(thresholds["tracking_error_high"])
        hot_out = states[:, 0]
        return {
            "outlet_temperature_above_limit": bool(np.any(hot_out > high)),
            "outlet_temperature_below_limit": bool(np.any(hot_out < low)),
            "large_temperature_tracking_error": bool(np.any(np.abs(hot_out - target) > tracking)),
        }
    raise ValueError(f"unknown system_id for events: {system_id}")


def trajectory_bad_event(system_id: str, states: np.ndarray, event_config: dict[str, Any]) -> bool:
    return bool(any(trajectory_event_flags(system_id, states, event_config).values()))


def _event_summary_for_dataset(dataset: dict[str, TrajectoryBatch], event_config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for split, batch in dataset.items():
        labels = [trajectory_bad_event(batch.system_id, batch.states[idx], event_config) for idx in range(batch.n_trajectories)]
        rows.append(
            {
                "system_id": batch.system_id,
                "split": split,
                "role": _split_role(split),
                "scenario_type": ",".join(sorted(set(batch.scenario_type))),
                "n": batch.n_trajectories,
                "event_positive_count": int(np.sum(labels)),
                "event_positive_rate": float(np.mean(labels)),
                "is_degenerate": bool(len(set(labels)) <= 1),
            }
        )
    return pd.DataFrame(rows)


def run_heat_exchanger_sanity(config_path: str | Path, output: str | Path) -> dict[str, Any]:
    config = load_v2_config(config_path)
    event_config = {"heat_exchanger": {"thresholds": config["heat_exchanger_event_thresholds"]}}
    out_dir = Path(output)
    _ensure_dir(out_dir)
    dataset = generate_v2_dataset("heat_exchanger", config, seed=0)
    data_summary = summarize_dataset(dataset)
    _write_json(out_dir / "data_summary.json", data_summary)

    distribution_rows = []
    id_batch = dataset["judge_test_id"]
    id_action_range = float(np.ptp(id_batch.actions))
    id_hot_range = float(np.ptp(id_batch.disturbances[..., 0]))
    for split, batch in dataset.items():
        if split == "model_train":
            continue
        action_range = float(np.ptp(batch.actions))
        hot_range = float(np.ptp(batch.disturbances[..., 0]))
        cold_range = float(np.ptp(batch.disturbances[..., 1]))
        distribution_rows.append(
            {
                "split": split,
                "role": _split_role(split),
                "scenario_type": ",".join(sorted(set(batch.scenario_type))),
                "action_range": action_range,
                "hot_inlet_range": hot_range,
                "cold_inlet_range": cold_range,
                "ood_differs_from_id": bool(
                    _split_role(split) == "judge_test"
                    and split != "judge_test_id"
                    and (action_range > id_action_range * 1.15 or hot_range > id_hot_range * 1.15)
                ),
            }
        )
    distribution = pd.DataFrame(distribution_rows)
    _write_csv(out_dir / "distribution_checks.csv", distribution)

    train = dataset["model_train"]
    model_rows = []
    for model_id in ["hold_last", "linear_narx"]:
        model = _make_v2_model(model_id, 0, config)
        model.fit(train)
        for split, batch in dataset.items():
            if _split_role(split) not in {"judge_calibration", "judge_test"}:
                continue
            errors = []
            for idx in range(batch.n_trajectories):
                pred = model.predict_rollout(batch.states[idx, 0], batch.actions[idx], batch.disturbances[idx])
                errors.append(rmse(pred, batch.states[idx]))
            model_rows.append(
                {
                    "model_id": model_id,
                    "split": split,
                    "role": _split_role(split),
                    "scenario_type": ",".join(sorted(set(batch.scenario_type))),
                    "rmse_mean": float(np.mean(errors)),
                    "rmse_min": float(np.min(errors)),
                    "rmse_max": float(np.max(errors)),
                }
            )
    model_errors = pd.DataFrame(model_rows)
    _write_csv(out_dir / "model_error_checks.csv", model_errors)

    event_checks = _event_summary_for_dataset(dataset, event_config)
    test_events = event_checks[event_checks["role"] == "judge_test"]
    event_non_degenerate = bool(test_events["event_positive_rate"].between(0.0, 1.0, inclusive="neither").any())
    linear = model_errors[model_errors["model_id"] == "linear_narx"]
    id_error = float(linear[linear["split"] == "judge_test_id"]["rmse_mean"].mean())
    ood_error = float(linear[(linear["role"] == "judge_test") & (linear["split"] != "judge_test_id")]["rmse_mean"].mean())
    bad_labels = model_errors["rmse_mean"] > float(config["rmse_thresholds"][2])
    bad_rmse_non_degenerate = bool(bad_labels.nunique() > 1)
    overlap = split_overlap_report(dataset)
    finite = all(np.isfinite(batch.states).all() and np.isfinite(batch.actions).all() and np.isfinite(batch.disturbances).all() for batch in dataset.values())
    nonconstant = all(
        float(np.max(np.std(batch.states.reshape(-1, batch.states.shape[-1]), axis=0))) > 1e-8
        for batch in dataset.values()
    )
    ood_differs = bool(distribution["ood_differs_from_id"].any())

    if finite and nonconstant and ood_differs and ood_error > id_error and bad_rmse_non_degenerate and overlap["overlap_count"] == 0:
        verdict = "VALID_HEAT_EXCHANGER_BENCHMARK"
    elif finite and nonconstant and ood_differs:
        verdict = "WEAK_HEAT_EXCHANGER_BENCHMARK"
    else:
        verdict = "INVALID_HEAT_EXCHANGER_BENCHMARK"
    event_payload = {
        "system_id": "heat_exchanger",
        "thresholds": config["heat_exchanger_event_thresholds"],
        "event_labels_explicit": True,
        "event_labels_non_degenerate": event_non_degenerate,
        "bad_rmse_labels_non_degenerate": bad_rmse_non_degenerate,
        "id_linear_rmse_mean": id_error,
        "ood_linear_rmse_mean": ood_error,
        "calibration_test_overlap": overlap,
        "finite": finite,
        "nonconstant": nonconstant,
        "ood_differs_from_id": ood_differs,
        "verdict": verdict,
    }
    _write_json(out_dir / "event_label_checks.json", event_payload)
    report = f"""# v2 Heat Exchanger Sanity

## Verdict

{verdict}

## Checks

- finite trajectories: {finite}
- nonconstant trajectories: {nonconstant}
- OOD differs from ID: {ood_differs}
- OOD linear RMSE > ID linear RMSE: {ood_error > id_error}
- bad RMSE labels non-degenerate: {bad_rmse_non_degenerate}
- event labels non-degenerate: {event_non_degenerate}
- calibration/test overlap count: {overlap["overlap_count"]}

## Distribution Checks

{_markdown_table(distribution, ["split", "scenario_type", "action_range", "hot_inlet_range", "ood_differs_from_id"], max_rows=12)}

## Model Error Checks

{_markdown_table(model_errors, ["model_id", "split", "rmse_mean"], max_rows=12)}
"""
    Path("reports/v2_heat_exchanger_sanity.md").write_text(report, encoding="utf-8")
    return event_payload


def validate_event_targets(config_path: str | Path, event_config_path: str | Path, output: str | Path) -> dict[str, Any]:
    config = load_v2_config(config_path)
    event_config = load_event_config(event_config_path)
    out_dir = Path(output)
    _ensure_dir(out_dir)
    frames = []
    for system_id in config["systems"]:
        dataset = generate_v2_dataset(system_id, config, seed=7)
        frame = _event_summary_for_dataset(dataset, event_config)
        frames.append(frame)
    validation = pd.concat(frames, ignore_index=True)
    _write_csv(out_dir / "event_target_validation.csv", validation)
    by_system = []
    for system_id, system_df in validation.groupby("system_id"):
        test_df = system_df[system_df["role"] == "judge_test"]
        non_degenerate = bool(test_df["event_positive_rate"].between(0.0, 1.0, inclusive="neither").any())
        by_system.append(
            {
                "system_id": system_id,
                "event_labels_non_degenerate": non_degenerate,
                "event_positive_rate_min": float(test_df["event_positive_rate"].min()),
                "event_positive_rate_max": float(test_df["event_positive_rate"].max()),
                "thresholds": event_config[system_id]["thresholds"],
            }
        )
    nondegenerate_count = sum(row["event_labels_non_degenerate"] for row in by_system)
    verdict = "EVENT_TARGETS_VALID" if nondegenerate_count >= 2 else "EVENT_TARGETS_WEAK"
    summary = {
        "verdict": verdict,
        "systems_checked": list(config["systems"]),
        "event_labels_from_trajectories": True,
        "nondegenerate_system_count": int(nondegenerate_count),
        "by_system": by_system,
    }
    _write_json(out_dir / "event_target_summary.json", summary)
    report = f"""# v2 Event Target Validation

## Verdict

{verdict}

## Summary

Non-degenerate systems: {nondegenerate_count}

## By System

{_markdown_table(pd.DataFrame(by_system), ["system_id", "event_labels_non_degenerate", "event_positive_rate_min", "event_positive_rate_max"])}
"""
    Path("reports/v2_event_target_validation.md").write_text(report, encoding="utf-8")
    return summary


def _prediction_rows_for_seed_system(
    system_id: str,
    seed: int,
    config: dict[str, Any],
    event_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    dataset = generate_v2_dataset(system_id, config, seed=seed)
    overlap = split_overlap_report(dataset)
    if overlap["overlap_count"] > 0:
        raise RuntimeError(f"calibration/test overlap detected for {system_id} seed {seed}")
    system = make_system(system_id)
    dt = float(config["data"]["dt"])
    support = SupportDistance()
    support.fit(dataset["model_train"])
    models = {}
    predictions: dict[tuple[str, str, int], np.ndarray] = {}
    for model_id in config["models"]:
        model = _make_v2_model(model_id, seed, config)
        model.fit(dataset["model_train"])
        models[model_id] = model
        for split, batch in dataset.items():
            if _split_role(split) not in {"judge_calibration", "judge_test"}:
                continue
            for idx in range(batch.n_trajectories):
                predictions[(model_id, split, idx)] = model.predict_rollout(
                    batch.states[idx, 0], batch.actions[idx], batch.disturbances[idx]
                )
    rows = []
    for split, batch in dataset.items():
        role = _split_role(split)
        if role not in {"judge_calibration", "judge_test"}:
            continue
        for idx in range(batch.n_trajectories):
            true_states = batch.states[idx]
            support_score = support.score(batch.actions[idx], batch.disturbances[idx])
            scenario_predictions = [predictions[(model_id, split, idx)] for model_id in config["models"]]
            scenario_disagreement = disagreement_score(scenario_predictions)
            true_event = trajectory_bad_event(system_id, true_states, event_config)
            for model_id, model in models.items():
                pred = predictions[(model_id, split, idx)]
                pred_event = trajectory_bad_event(system_id, pred, event_config)
                try:
                    uncertainty = uncertainty_score(
                        model,
                        batch.states[idx, 0],
                        batch.actions[idx],
                        batch.disturbances[idx],
                        n_samples=int(config["data"].get("uncertainty_samples", 2)),
                    )
                except Exception:
                    uncertainty = 0.0
                rows.append(
                    {
                        "system_id": system_id,
                        "seed": seed,
                        "split": split,
                        "role": role,
                        "scenario_type": batch.scenario_type[idx],
                        "scenario_index": idx,
                        "scenario_id": f"{system_id}_seed{seed}_{split}_{idx:04d}",
                        "model_id": model_id,
                        "rmse": rmse(pred, true_states),
                        "mae": mae(pred, true_states),
                        "max_abs_error": max_abs_error(pred, true_states),
                        "final_state_error": final_state_error(pred, true_states),
                        "true_event": bool(true_event),
                        "pred_event": bool(pred_event),
                        "event_mismatch": bool(true_event != pred_event),
                        "support_distance": support_score,
                        "uncertainty_score": float(uncertainty),
                        "disagreement_score": scenario_disagreement,
                        "invariant_residual": invariant_residual_score(system, pred, batch.actions[idx], batch.disturbances[idx], dt),
                        "repair_amount": repair_amount_score(system, pred),
                    }
                )
    score_frame = pd.DataFrame(rows)
    model_metrics = (
        score_frame[score_frame["role"] == "judge_test"]
        .groupby(["system_id", "seed", "model_id", "split"], as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            mae_mean=("mae", "mean"),
            max_abs_error_mean=("max_abs_error", "mean"),
            final_state_error_mean=("final_state_error", "mean"),
            n_scenarios=("scenario_id", "count"),
        )
    )
    return score_frame, model_metrics, {"split_overlap": overlap}


def _target_rows(base: pd.DataFrame, target: str, threshold: float | None) -> pd.DataFrame:
    frame = base.copy()
    if target == "bad_rmse":
        if threshold is None:
            raise ValueError("bad_rmse requires threshold")
        frame["badness_error"] = frame["rmse"].astype(float)
        frame["bad_label"] = frame["rmse"].astype(float) > float(threshold)
        frame["bad_threshold"] = float(threshold)
    elif target == "bad_event":
        frame["badness_error"] = frame["event_mismatch"].astype(float)
        frame["bad_label"] = frame["event_mismatch"].astype(bool)
        frame["bad_threshold"] = 0.5
    elif target == "bad_rmse_or_event":
        if threshold is None:
            raise ValueError("bad_rmse_or_event requires threshold")
        frame["badness_error"] = ((frame["rmse"].astype(float) > float(threshold)) | frame["event_mismatch"].astype(bool)).astype(float)
        frame["bad_label"] = frame["badness_error"].astype(bool)
        frame["bad_threshold"] = float(threshold)
    else:
        raise ValueError(f"unknown badness target: {target}")
    return frame


def _fit_stronger_baselines(calibration: pd.DataFrame, primary_coverages: list[float]) -> dict[str, Any]:
    labels = calibration["bad_label"].astype(int)
    baselines: dict[str, Any] = {}
    if labels.nunique() >= 2:
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced"))
        model.fit(calibration[SIGNAL_COLUMNS], labels)
        baselines["learned_error_classifier"] = model
    else:
        baselines["learned_error_classifier"] = None

    errors = calibration["badness_error"].to_numpy(dtype=float)
    best = None
    for signal in SIGNAL_COLUMNS:
        for orientation in [1, -1]:
            score = orientation * calibration[signal].to_numpy(dtype=float)
            objective = []
            for coverage in primary_coverages:
                accepted = np.argsort(score, kind="mergesort")[: max(1, int(math.ceil(coverage * len(score))))]
                objective.append(float(np.mean(calibration["bad_label"].to_numpy(dtype=bool)[accepted])))
            low_count = max(1, int(math.ceil(min(primary_coverages) * len(score))))
            low_accept = np.argsort(score, kind="mergesort")[:low_count]
            candidate = (
                float(np.mean(objective)),
                float(np.mean(errors[low_accept])),
                signal,
                orientation,
            )
            if best is None or candidate < best:
                best = candidate
    baselines["conformal_risk_threshold"] = {"signal": best[2], "orientation": best[3]} if best is not None else None
    baselines["ensemble_disagreement_threshold"] = {"signal": "disagreement_score", "orientation": 1}
    return baselines


def _low_coverage_far(labels: np.ndarray, score: np.ndarray, coverages: list[float]) -> float:
    labels = np.asarray(labels, dtype=bool)
    score = np.asarray(score, dtype=float)
    order = np.argsort(score, kind="mergesort")
    fars = []
    for coverage in coverages:
        count = min(max(int(math.ceil(float(coverage) * len(labels))), 1), len(labels))
        fars.append(float(np.mean(labels[order[:count]])))
    return float(np.mean(fars))


def _signal_orientation(calibration: pd.DataFrame, signal: str) -> int:
    labels = calibration["bad_label"].astype(bool).to_numpy()
    values = calibration[signal].to_numpy(dtype=float)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return 1
    bad_mean = float(np.mean(values[labels]))
    good_mean = float(np.mean(values[~labels]))
    return 1 if bad_mean >= good_mean else -1


def _best_signal_params(calibration: pd.DataFrame, primary_coverages: list[float]) -> dict[str, Any]:
    labels = calibration["bad_label"].astype(bool).to_numpy()
    candidates = []
    for signal in SIGNAL_COLUMNS:
        orientation = _signal_orientation(calibration, signal)
        score = orientation * calibration[signal].to_numpy(dtype=float)
        candidates.append(
            {
                "signal": signal,
                "orientation": orientation,
                "objective": _low_coverage_far(labels, score, primary_coverages),
                "mean_score": float(np.mean(score)),
            }
        )
    return sorted(candidates, key=lambda row: (row["objective"], row["mean_score"], row["signal"]))[0]


def _rank_mean_score(calibration: pd.DataFrame, table: pd.DataFrame, orientations: dict[str, int]) -> np.ndarray:
    scores = []
    for signal in SIGNAL_COLUMNS:
        reference = np.sort(orientations[signal] * calibration[signal].to_numpy(dtype=float))
        values = orientations[signal] * table[signal].to_numpy(dtype=float)
        scores.append(np.searchsorted(reference, values, side="right") / max(len(reference), 1))
    return np.mean(np.vstack(scores), axis=0)


def _fit_v2_calibrated_scores(calibration: pd.DataFrame, test: pd.DataFrame, primary_coverages: list[float]) -> dict[str, np.ndarray]:
    labels = calibration["bad_label"].astype(bool).to_numpy()
    orientations = {signal: _signal_orientation(calibration, signal) for signal in SIGNAL_COLUMNS}
    best_signal = _best_signal_params(calibration, primary_coverages)
    best_score_test = int(best_signal["orientation"]) * test[str(best_signal["signal"])].to_numpy(dtype=float)
    rank_mean_test = _rank_mean_score(calibration, test, orientations)
    natural_sum = np.zeros(len(test), dtype=float)
    for signal in SIGNAL_COLUMNS:
        values = test[signal].to_numpy(dtype=float)
        cal_values = calibration[signal].to_numpy(dtype=float)
        denom = max(float(np.max(cal_values) - np.min(cal_values)), 1e-12)
        natural_sum += (values - float(np.min(cal_values))) / denom
    natural_sum /= len(SIGNAL_COLUMNS)
    if labels.sum() > 0 and labels.sum() < len(labels):
        logistic = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, class_weight="balanced"))
        logistic.fit(calibration[SIGNAL_COLUMNS], labels.astype(int))
        logistic_score = logistic.predict_proba(test[SIGNAL_COLUMNS])[:, 1]
    else:
        logistic_score = rank_mean_test
    cal_best_score = int(best_signal["orientation"]) * calibration[str(best_signal["signal"])].to_numpy(dtype=float)
    reference = np.sort(cal_best_score)
    isotonic_like = np.searchsorted(reference, best_score_test, side="right") / max(len(reference), 1)
    thresholds = {
        signal: float(np.quantile(calibration[signal].to_numpy(dtype=float), 0.20))
        for signal in SIGNAL_COLUMNS
    }
    conservative_parts = []
    for signal in SIGNAL_COLUMNS:
        values = test[signal].to_numpy(dtype=float)
        scale = max(float(np.std(calibration[signal].to_numpy(dtype=float))), 1e-6)
        conservative_parts.append(np.maximum((values - thresholds[signal]) / scale, 0.0))
    conservative = np.mean(np.vstack(conservative_parts), axis=0)
    candidates = {
        "best_signal": best_score_test,
        "rank_mean": rank_mean_test,
        "natural_sum": natural_sum,
        "logistic": logistic_score,
    }
    cal_candidate_scores = {
        "best_signal": int(best_signal["orientation"]) * calibration[str(best_signal["signal"])].to_numpy(dtype=float),
        "rank_mean": _rank_mean_score(calibration, calibration, orientations),
        "natural_sum": np.mean(
            np.vstack(
                [
                    (calibration[signal].to_numpy(dtype=float) - float(np.min(calibration[signal].to_numpy(dtype=float))))
                    / max(float(np.ptp(calibration[signal].to_numpy(dtype=float))), 1e-12)
                    for signal in SIGNAL_COLUMNS
                ]
            ),
            axis=0,
        ),
        "logistic": logistic.predict_proba(calibration[SIGNAL_COLUMNS])[:, 1]
        if labels.sum() > 0 and labels.sum() < len(labels)
        else _rank_mean_score(calibration, calibration, orientations),
    }
    selected_candidate = min(
        cal_candidate_scores,
        key=lambda name: _low_coverage_far(labels, cal_candidate_scores[name], primary_coverages),
    )
    return {
        "best_single_signal_selected_on_calibration": best_score_test,
        "calibration_selected_candidate_ranker": candidates[selected_candidate],
        "rank_normalized_linear": rank_mean_test,
        "logistic_calibrated_judge": logistic_score,
        "isotonic_calibrated_judge": isotonic_like,
        "quantile_rule_judge": best_score_test,
        "conservative_low_coverage_judge": conservative,
    }


def _score_stronger_baseline(name: str, fitted: Any, table: pd.DataFrame) -> np.ndarray:
    if name == "learned_error_classifier":
        if fitted is None:
            return np.ones(len(table), dtype=float)
        return fitted.predict_proba(table[SIGNAL_COLUMNS])[:, 1]
    if name == "conformal_risk_threshold":
        if fitted is None:
            return table["support_distance"].to_numpy(dtype=float)
        return int(fitted["orientation"]) * table[str(fitted["signal"])].to_numpy(dtype=float)
    if name == "ensemble_disagreement_threshold":
        return table["disagreement_score"].to_numpy(dtype=float)
    raise ValueError(f"unknown stronger baseline: {name}")


def _risk_curve_from_labels(labels: np.ndarray, risk_scores: np.ndarray, coverages: list[float]) -> pd.DataFrame:
    labels = np.asarray(labels, dtype=bool)
    risk_scores = np.asarray(risk_scores, dtype=float)
    if len(labels) == 0:
        raise ValueError("at least one scenario required")
    if not np.isfinite(risk_scores).all():
        raise ValueError("risk scores must be finite")
    order = np.argsort(risk_scores, kind="mergesort")
    sorted_labels = labels[order]
    rows = []
    n = len(labels)
    for coverage in coverages:
        accepted_count = min(max(int(math.ceil(float(coverage) * n)), 1), n)
        accepted = sorted_labels[:accepted_count]
        false_accept_count = int(np.sum(accepted))
        rows.append(
            {
                "coverage": float(coverage),
                "accepted_count": accepted_count,
                "false_accept_count": false_accept_count,
                "false_accept_rate": float(false_accept_count / accepted_count),
            }
        )
    return pd.DataFrame(rows)


def _compute_risk_coverage_for_target(
    frame: pd.DataFrame,
    config: dict[str, Any],
    target: str,
    threshold: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_frame = _target_rows(frame, target, threshold)
    calibration = target_frame[target_frame["role"] == "judge_calibration"].copy()
    test = target_frame[target_frame["role"] == "judge_test"].copy()
    risk_frames = []
    scenario_frames = []
    primary = [float(value) for value in config["primary_coverages"]]

    fitted_calibrated = []
    stronger = _fit_stronger_baselines(calibration, primary)

    for judge_id in config["judges"]["simple"]:
        if judge_id == "random_baseline":
            risk = np.random.default_rng(1000 + int(test["seed"].iloc[0])).uniform(0.0, 1.0, len(test))
        else:
            risk = test[SIMPLE_RISK_COLUMN[judge_id]].to_numpy(dtype=float)
        test[f"risk_{judge_id}"] = risk
    for name, fitted in stronger.items():
        test[f"risk_{name}"] = _score_stronger_baseline(name, fitted, test)
    calibrated_scores = _fit_v2_calibrated_scores(calibration, test, primary)
    for judge_id, score in calibrated_scores.items():
        test[f"risk_{judge_id}"] = score
    test["risk_oracle_error_rank"] = test["badness_error"].to_numpy(dtype=float)
    scenario_frames.append(
        test[
            [
                "system_id",
                "seed",
                "split",
                "scenario_type",
                "scenario_id",
                "model_id",
                "rmse",
                "true_event",
                "pred_event",
                "event_mismatch",
                "bad_label",
                "bad_threshold",
                *SIGNAL_COLUMNS,
                *[column for column in test.columns if column.startswith("risk_")],
            ]
        ].assign(badness_target=target)
    )

    all_judges = [
        *config["judges"]["simple"],
        *config["judges"]["stronger_baselines"],
        *config["judges"]["calibrated"],
        *config["judges"]["diagnostic_only"],
    ]
    for model_id, model_test in test.groupby("model_id", sort=False):
        labels = model_test["bad_label"].to_numpy(dtype=bool)
        for judge_id in all_judges:
            curve = _risk_curve_from_labels(labels, model_test[f"risk_{judge_id}"].to_numpy(dtype=float), config["coverage_grid"])
            curve["system_id"] = str(model_test["system_id"].iloc[0])
            curve["seed"] = int(model_test["seed"].iloc[0])
            curve["model_id"] = model_id
            curve["badness_target"] = target
            curve["bad_threshold"] = float(model_test["bad_threshold"].iloc[0])
            curve["judge_id"] = judge_id
            curve["is_oracle"] = judge_id == "oracle_error_rank"
            curve["is_real_judge"] = judge_id != "oracle_error_rank"
            curve["is_calibrated"] = judge_id in config["judges"]["calibrated"]
            risk_frames.append(curve)

    risk = pd.concat(risk_frames, ignore_index=True)
    baseline_ids = [*config["judges"]["simple"], *config["judges"]["stronger_baselines"]]
    baseline = (
        risk[risk["judge_id"].isin(baseline_ids)]
        .sort_values(["false_accept_rate", "judge_id"], ascending=[True, True])
        .groupby(["system_id", "seed", "model_id", "badness_target", "bad_threshold", "coverage"], as_index=False)
        .first()[["system_id", "seed", "model_id", "badness_target", "bad_threshold", "coverage", "judge_id", "false_accept_rate"]]
        .rename(columns={"judge_id": "baseline_judge", "false_accept_rate": "baseline_far"})
    )
    risk = risk.merge(
        baseline,
        on=["system_id", "seed", "model_id", "badness_target", "bad_threshold", "coverage"],
        how="left",
    )
    risk["absolute_margin"] = risk["baseline_far"] - risk["false_accept_rate"]
    risk["relative_margin"] = np.where(
        risk["baseline_far"] > 0.0,
        risk["absolute_margin"] / risk["baseline_far"],
        0.0,
    )
    oracle = risk[risk["judge_id"] == "oracle_error_rank"][
        ["system_id", "seed", "model_id", "badness_target", "bad_threshold", "coverage", "false_accept_rate"]
    ].rename(columns={"false_accept_rate": "oracle_far"})
    primary = risk[risk["judge_id"] == str(config.get("primary_calibrated_judge", "calibration_selected_candidate_ranker"))][
        ["system_id", "seed", "model_id", "badness_target", "bad_threshold", "coverage", "false_accept_rate"]
    ].rename(columns={"false_accept_rate": "primary_calibrated_far"})
    oracle_gap = primary.merge(
        oracle,
        on=["system_id", "seed", "model_id", "badness_target", "bad_threshold", "coverage"],
        how="left",
    )
    oracle_gap["oracle_gap"] = oracle_gap["primary_calibrated_far"] - oracle_gap["oracle_far"]
    return risk, pd.concat(scenario_frames, ignore_index=True), oracle_gap


def _valid_systems_for_v2(config: dict[str, Any], output: Path) -> list[str]:
    systems = list(config["systems"])
    if "heat_exchanger" in systems:
        sanity_path = output.parent / "heat_exchanger_sanity" / "event_label_checks.json"
        if not sanity_path.exists():
            raise RuntimeError("heat exchanger sanity must run before v2 frozen protocol")
        sanity = _load_json(sanity_path)
        if sanity.get("verdict") != "VALID_HEAT_EXCHANGER_BENCHMARK":
            systems = [system for system in systems if system != "heat_exchanger"]
    return systems


def run_v2_frozen_protocol(config_path: str | Path, event_config_path: str | Path, output: str | Path) -> dict[str, Any]:
    config = load_v2_config(config_path)
    event_config = load_event_config(event_config_path)
    out_dir = Path(output)
    _ensure_dir(out_dir)
    valid_systems = _valid_systems_for_v2(config, out_dir)
    score_frames = []
    metric_frames = []
    risk_frames = []
    scenario_risk_frames = []
    oracle_frames = []
    leakage = []
    for system_id in valid_systems:
        for seed in config["seeds"]:
            base_scores, model_metrics, diagnostics = _prediction_rows_for_seed_system(system_id, int(seed), config, event_config)
            leakage.append({"system_id": system_id, "seed": int(seed), **diagnostics["split_overlap"]})
            score_frames.append(base_scores)
            metric_frames.append(model_metrics)
            for target in config["badness_targets"]:
                thresholds: list[float | None] = [None] if target == "bad_event" else [float(value) for value in config["rmse_thresholds"]]
                for threshold in thresholds:
                    risk, scenario_risk, oracle = _compute_risk_coverage_for_target(base_scores, config, target, threshold)
                    risk_frames.append(risk)
                    scenario_risk_frames.append(scenario_risk)
                    oracle_frames.append(oracle)
    scenario_scores = pd.concat(score_frames, ignore_index=True)
    model_metrics = pd.concat(metric_frames, ignore_index=True)
    risk_coverage = pd.concat(risk_frames, ignore_index=True)
    scenario_risks = pd.concat(scenario_risk_frames, ignore_index=True)
    oracle_gap = pd.concat(oracle_frames, ignore_index=True)
    event_metrics = (
        scenario_scores[scenario_scores["role"] == "judge_test"]
        .groupby(["system_id", "seed", "model_id", "split"], as_index=False)
        .agg(
            true_event_rate=("true_event", "mean"),
            pred_event_rate=("pred_event", "mean"),
            event_mismatch_rate=("event_mismatch", "mean"),
            n_scenarios=("scenario_id", "count"),
        )
    )
    for name, frame in [
        ("v2_scenario_scores.csv", scenario_risks),
        ("v2_risk_coverage.csv", risk_coverage),
        ("v2_model_metrics.csv", model_metrics),
        ("v2_event_metrics.csv", event_metrics),
        ("v2_oracle_gap.csv", oracle_gap),
    ]:
        if frame.empty or frame.isna().any().any():
            raise RuntimeError(f"{name} is empty or contains NaN")
        _write_csv(out_dir / name, frame)
    leakage_detected = any(row["overlap_count"] > 0 for row in leakage)
    summary = {
        "verdict": "V2_FROZEN_PROTOCOL_COMPLETE" if not leakage_detected else "V2_FROZEN_PROTOCOL_INVALID_LEAKAGE",
        "valid_systems": valid_systems,
        "attempted_systems": list(config["systems"]),
        "models_evaluated": list(config["models"]),
        "badness_targets": list(config["badness_targets"]),
        "seed_count": len(config["seeds"]),
        "seeds": list(config["seeds"]),
        "coverage_grid": list(config["coverage_grid"]),
        "rmse_thresholds": list(config["rmse_thresholds"]),
        "leakage_detected": leakage_detected,
        "split_overlap": leakage,
        "row_counts": {
            "scenario_scores": int(len(scenario_risks)),
            "risk_coverage": int(len(risk_coverage)),
            "model_metrics": int(len(model_metrics)),
            "event_metrics": int(len(event_metrics)),
            "oracle_gap": int(len(oracle_gap)),
        },
        "protocol_hash": _protocol_hash(),
    }
    _write_json(out_dir / "v2_run_summary.json", summary)
    report = f"""# v2 Frozen Protocol Report

## Verdict

{summary["verdict"]}

## Valid Systems

{", ".join(valid_systems)}

## Models

{", ".join(config["models"])}

## Badness Targets

{", ".join(config["badness_targets"])}

## Risk Coverage Preview

{_markdown_table(risk_coverage.head(12), ["system_id", "seed", "model_id", "badness_target", "judge_id", "coverage", "false_accept_rate", "baseline_judge", "absolute_margin"])}

## Leakage

Detected: {leakage_detected}
"""
    Path("reports/v2_frozen_protocol_report.md").write_text(report, encoding="utf-8")
    return summary


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_samples: int = 500) -> tuple[float, float, np.ndarray]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return 0.0, 0.0, np.zeros(n_samples, dtype=float)
    samples = np.empty(n_samples, dtype=float)
    for idx in range(n_samples):
        samples[idx] = float(np.mean(rng.choice(values, size=len(values), replace=True)))
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975)), samples


def run_v2_statistical_audit(config_path: str | Path, results: str | Path, output: str | Path) -> dict[str, Any]:
    config = load_v2_config(config_path)
    results_dir = Path(results)
    out_dir = Path(output)
    _ensure_dir(out_dir)
    risk = pd.read_csv(results_dir / "v2_risk_coverage.csv")
    event_metrics = pd.read_csv(results_dir / "v2_event_metrics.csv")
    summary_run = _load_json(results_dir / "v2_run_summary.json")
    primary_judge = str(config.get("primary_calibrated_judge", "calibration_selected_candidate_ranker"))
    primary_coverages = [float(value) for value in config["primary_coverages"]]
    primary = risk[(risk["judge_id"] == primary_judge) & (risk["coverage"].isin(primary_coverages))].copy()
    primary["seed_win"] = primary["absolute_margin"] > 0.0
    seed_level = (
        primary.groupby(["system_id", "seed", "badness_target", "coverage"], as_index=False)
        .agg(
            mean_absolute_margin=("absolute_margin", "mean"),
            mean_relative_margin=("relative_margin", "mean"),
            seed_win=("seed_win", "mean"),
            false_accept_rate=("false_accept_rate", "mean"),
            baseline_far=("baseline_far", "mean"),
        )
    )
    _write_csv(out_dir / "v2_seed_level_margins.csv", seed_level)

    rng = np.random.default_rng(2025)
    effect_rows = []
    bootstrap_rows = []
    thresholds = config["practical_thresholds"]
    for (system_id, target), group in seed_level.groupby(["system_id", "badness_target"], sort=False):
        values = group["mean_absolute_margin"].to_numpy(dtype=float)
        ci_low, ci_high, samples = _bootstrap_ci(values, rng)
        win_rate = float(np.mean(values > 0.0))
        mean_margin = float(np.mean(values))
        mean_rel = float(group["mean_relative_margin"].mean())
        practical = bool(
            mean_margin >= float(thresholds["minimum_absolute_far_reduction"])
            and mean_rel >= float(thresholds["minimum_relative_far_reduction"])
            and win_rate >= float(thresholds["minimum_seed_win_rate_strong"])
        )
        event_worsening = bool(target == "bad_event" and mean_margin < 0.0)
        effect_rows.append(
            {
                "system_id": system_id,
                "badness_target": target,
                "mean_far_margin": mean_margin,
                "relative_far_margin": mean_rel,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "seed_win_rate": win_rate,
                "practical_threshold_pass": practical,
                "ci_excludes_zero": bool(ci_low > 0.0 or ci_high < 0.0),
                "positive_ci_excludes_zero": bool(ci_low > 0.0),
                "event_risk_worsening": event_worsening,
                "accepted_catastrophic_failure_count": int(
                    risk[
                        (risk["system_id"] == system_id)
                        & (risk["badness_target"] == target)
                        & (risk["judge_id"] == primary_judge)
                        & (risk["coverage"].isin(primary_coverages))
                    ]["false_accept_count"].sum()
                ),
            }
        )
        for sample_idx, value in enumerate(samples):
            bootstrap_rows.append(
                {
                    "system_id": system_id,
                    "badness_target": target,
                    "sample_index": sample_idx,
                    "mean_far_margin": float(value),
                }
            )
    effect = pd.DataFrame(effect_rows)
    bootstrap = pd.DataFrame(bootstrap_rows)
    _write_csv(out_dir / "v2_effect_size_by_system.csv", effect)
    _write_csv(out_dir / "v2_bootstrap_samples.csv", bootstrap)

    auc_rows = []
    for keys, group in risk[risk["judge_id"] == primary_judge].groupby(["system_id", "badness_target", "model_id", "seed"], sort=False):
        ordered = group.sort_values("coverage")
        auc_rows.append(
            {
                "system_id": keys[0],
                "badness_target": keys[1],
                "model_id": keys[2],
                "seed": keys[3],
                "risk_coverage_auc": float(np.trapezoid(ordered["false_accept_rate"], ordered["coverage"])),
            }
        )
    auc = pd.DataFrame(auc_rows)
    event_worsening = bool(effect["event_risk_worsening"].any())
    leakage = bool(summary_run.get("leakage_detected", False))
    valid_systems = list(summary_run.get("valid_systems", []))
    positive_systems = sorted(effect.groupby("system_id")["mean_far_margin"].mean().loc[lambda s: s > 0.0].index.tolist())
    practical_systems = sorted(effect.groupby("system_id")["practical_threshold_pass"].any().loc[lambda s: s].index.tolist())
    ci_positive_systems = sorted(effect.groupby("system_id")["positive_ci_excludes_zero"].any().loc[lambda s: s].index.tolist())
    if leakage or len(valid_systems) < 2:
        verdict = "INVALID_DUE_TO_LEAKAGE_OR_BENCHMARK_FAILURE"
    elif len(valid_systems) >= 3 and len(practical_systems) >= 3 and len(ci_positive_systems) >= 2 and not event_worsening:
        verdict = "STRONG_MULTI_SYSTEM_EFFECT"
    elif len(positive_systems) > 0 and len(positive_systems) < len(valid_systems):
        verdict = "MIXED_SYSTEM_DEPENDENT_EFFECT"
    elif len(positive_systems) == len(valid_systems) and len(positive_systems) > 0:
        verdict = "WEAK_MULTI_SYSTEM_EFFECT"
    else:
        verdict = "NO_ROBUST_EFFECT"
    statistical_summary = {
        "verdict": verdict,
        "valid_systems": valid_systems,
        "primary_calibrated_judge": primary_judge,
        "positive_systems": positive_systems,
        "practical_threshold_systems": practical_systems,
        "ci_positive_systems": ci_positive_systems,
        "event_risk_worsening": event_worsening,
        "leakage_detected": leakage,
        "effect_rows": effect.to_dict(orient="records"),
        "risk_coverage_auc_mean": float(auc["risk_coverage_auc"].mean()) if not auc.empty else 0.0,
        "event_metrics_rows": int(len(event_metrics)),
    }
    _write_json(out_dir / "v2_statistical_summary.json", statistical_summary)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot_frame = effect.groupby("system_id", as_index=False)["mean_far_margin"].mean()
    ax.bar(plot_frame["system_id"], plot_frame["mean_far_margin"], color="#4C78A8")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_ylabel("Mean FAR margin vs strongest baseline")
    ax.set_xlabel("System")
    ax.set_title("v2 low-coverage calibrated refusal effect")
    fig.tight_layout()
    fig.savefig(out_dir / "v2_effect_size_plot.png", dpi=160)
    plt.close(fig)
    report = f"""# v2 Statistical Audit

## Verdict

{verdict}

## Effect Size By System

{_markdown_table(effect, ["system_id", "badness_target", "mean_far_margin", "bootstrap_ci_low", "bootstrap_ci_high", "seed_win_rate", "practical_threshold_pass"], max_rows=18)}

## Event-Risk Check

Event-risk worsening: {event_worsening}

## Leakage

Leakage detected: {leakage}
"""
    Path("reports/v2_statistical_audit.md").write_text(report, encoding="utf-8")
    return statistical_summary


def decide_v2_claim(
    heat_sanity: dict[str, Any],
    frozen_run: dict[str, Any],
    stats: dict[str, Any],
) -> dict[str, Any]:
    valid_systems = list(frozen_run.get("valid_systems", []))
    leakage = bool(frozen_run.get("leakage_detected", False) or stats.get("leakage_detected", False))
    event_worsening = bool(stats.get("event_risk_worsening", False))
    positive_systems = list(stats.get("positive_systems", []))
    practical_systems = list(stats.get("practical_threshold_systems", []))
    ci_systems = list(stats.get("ci_positive_systems", []))
    heat_invalid_used = "heat_exchanger" in valid_systems and heat_sanity.get("verdict") != "VALID_HEAT_EXCHANGER_BENCHMARK"
    if leakage or heat_invalid_used or stats.get("verdict") == "INVALID_DUE_TO_LEAKAGE_OR_BENCHMARK_FAILURE":
        decision = "INVALID_V2_PROTOCOL"
        allowed_claim = V2_ALLOWED_V1_CLAIM
    elif (
        len(valid_systems) >= 3
        and len(positive_systems) >= 3
        and len(practical_systems) >= 2
        and len(ci_systems) >= 2
        and not event_worsening
    ):
        decision = "UPGRADE_TO_MODERATE_MULTI_SYSTEM_LOW_COVERAGE_CLAIM"
        allowed_claim = "Calibrated low-coverage refusal shows a moderate multi-system synthetic benchmark effect under the frozen v2 protocol."
    elif 0 < len(positive_systems) < len(valid_systems):
        decision = "SYSTEM_DEPENDENT_BENCHMARK_RESULT"
        allowed_claim = "The benchmark shows system-dependent low-coverage refusal behavior under the frozen v2 protocol."
    elif len(positive_systems) > 0:
        decision = "KEEP_WEAK_LOW_COVERAGE_BENCHMARK_CLAIM"
        allowed_claim = V2_ALLOWED_V1_CLAIM
    else:
        decision = "NO_METHOD_CLAIM_BENCHMARK_ONLY"
        allowed_claim = "This repository is a benchmark only; v2 does not support a calibrated-refusal method claim."
    return {
        "decision": decision,
        "allowed_claim": allowed_claim,
        "forbidden_claims": [
            "safety certification",
            "trusted simulator",
            "validated digital twin",
            "general simulator reliability",
            "high-coverage reliability",
            "product readiness",
        ],
        "scientific_claim_upgraded": decision == "UPGRADE_TO_MODERATE_MULTI_SYSTEM_LOW_COVERAGE_CLAIM",
    }


def make_v2_scientific_decision_gate(
    protocol: str | Path,
    heat_sanity: str | Path,
    frozen_run: str | Path,
    stats: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    protocol_text = Path(protocol).read_text(encoding="utf-8")
    heat = _load_json(heat_sanity)
    frozen = _load_json(frozen_run)
    statistical = _load_json(stats)
    decision = decide_v2_claim(heat, frozen, statistical)
    if decision["decision"] not in V2_REQUIRED_DECISIONS:
        raise RuntimeError(f"invalid v2 decision label: {decision['decision']}")
    report = f"""# v2 Scientific Decision Gate

## Starting v1 claim

{V2_ALLOWED_V1_CLAIM}

## v2 protocol status

Protocol hash: {_sha256_file(Path(protocol))}

Protocol contains forbidden post-result tuning rule: {"Forbidden changes after results" in protocol_text}

## Valid systems

{", ".join(frozen.get("valid_systems", []))}

## Models evaluated

{", ".join(frozen.get("models_evaluated", []))}

## Badness targets

{", ".join(frozen.get("badness_targets", []))}

## Statistical evidence

Statistical verdict: {statistical.get("verdict")}

Positive systems: {statistical.get("positive_systems")}

Practical-threshold systems: {statistical.get("practical_threshold_systems")}

CI-positive systems: {statistical.get("ci_positive_systems")}

## Event-risk evidence

Event-risk worsening: {statistical.get("event_risk_worsening")}

## Decision

{decision["decision"]}

## Allowed claim

{decision["allowed_claim"]}

## Forbidden claims

{chr(10).join(f"- {claim}" for claim in decision["forbidden_claims"])}

## Recommended next action

Keep v2 evidence separate from v1 unless a future protocol explicitly permits a claim update.
"""
    output_path = Path(output)
    output_path.write_text(report, encoding="utf-8")
    if output_path == Path("reports/v2_scientific_decision_gate.md"):
        write_v2_docs(decision, frozen, statistical)
    return decision


def write_v2_docs(decision: dict[str, Any], frozen: dict[str, Any], stats: dict[str, Any]) -> None:
    _ensure_dir(Path("docs/v2"))
    summary = f"""# v2 Scientific Summary

## Decision

{decision["decision"]}

## Allowed claim

{decision["allowed_claim"]}

## Valid systems

{", ".join(frozen.get("valid_systems", []))}

## Models evaluated

{", ".join(frozen.get("models_evaluated", []))}

## Statistical verdict

{stats.get("verdict")}

## Limitations

The v1 claim is preserved. Negative and weak system results remain part of the v2 evidence package.
"""
    Path("docs/v2/v2_scientific_summary.md").write_text(summary, encoding="utf-8")
    claim_rows = [
        ["calibrated refusal works generally", "not allowed unless decision upgrades", "v2 decision gate", decision["allowed_claim"] if decision["scientific_claim_upgraded"] else "Do not claim general reliability."],
        ["calibrated refusal works at low coverage", "decision-gated", "v2 statistical audit", decision["allowed_claim"]],
        ["event-risk refusal works", "limited by event-risk audit", "v2 event metrics and statistical audit", "Only describe observed event-risk result."],
        ["signals are universal", "not supported", "system-dependent signal audits", "Do not claim universal signals."],
        ["repair_amount is universal", "not supported", "repair signal semantics audit", "Repair amount is role-dependent."],
        ["benchmark exposes system dependence", "allowed", "v2 per-system evidence", "The benchmark exposes system-dependent refusal behavior."],
    ]
    lines = ["# v2 Claim Audit Table", "", "| Claim | Status | Evidence | Allowed wording |", "|---|---|---|---|"]
    lines.extend("| " + " | ".join(row) + " |" for row in claim_rows)
    Path("docs/v2/v2_claim_audit_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    repro = """# v2 Reproducibility Card

Run:

```bash
python scripts/v2_verify_preconditions.py --config configs/v2/v2_scientific_strengthening.yaml --output results/v2_scientific_strengthening/preconditions
python scripts/v2_run_heat_exchanger_sanity.py --config configs/v2/v2_scientific_strengthening.yaml --output results/v2_scientific_strengthening/heat_exchanger_sanity
python scripts/v2_validate_event_targets.py --config configs/v2/v2_scientific_strengthening.yaml --event-config configs/v2/v2_event_targets.yaml --output results/v2_scientific_strengthening/event_targets
python scripts/v2_run_frozen_protocol.py --config configs/v2/v2_scientific_strengthening.yaml --event-config configs/v2/v2_event_targets.yaml --output results/v2_scientific_strengthening/frozen_protocol
python scripts/v2_statistical_audit.py --config configs/v2/v2_scientific_strengthening.yaml --results results/v2_scientific_strengthening/frozen_protocol --output results/v2_scientific_strengthening/statistical_audit
python scripts/v2_make_scientific_decision_gate.py --protocol docs/v2/v2_scientific_protocol_lock.md --heat-sanity results/v2_scientific_strengthening/heat_exchanger_sanity/event_label_checks.json --frozen-run results/v2_scientific_strengthening/frozen_protocol/v2_run_summary.json --stats results/v2_scientific_strengthening/statistical_audit/v2_statistical_summary.json --output reports/v2_scientific_decision_gate.md
```
"""
    Path("docs/v2/v2_reproducibility_card.md").write_text(repro, encoding="utf-8")
    release = f"""# Release Note: v2 Scientific Strengthening

Decision: {decision["decision"]}

Scientific claim upgraded: {"YES" if decision["scientific_claim_upgraded"] else "NO"}

Allowed claim: {decision["allowed_claim"]}

This release keeps v2 artifacts separate from v1 artifacts.
"""
    Path("reports/release_note_v2_scientific_strengthening.md").write_text(release, encoding="utf-8")


def _safe_auc(labels: pd.Series, scores: pd.Series) -> float:
    from sklearn.metrics import roc_auc_score

    y_true = labels.astype(int)
    y_score = scores.astype(float)
    if y_true.nunique() < 2 or y_score.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def diagnose_v2_calibrated_underperformance(results: str | Path, output: str | Path) -> dict[str, Any]:
    """Diagnose why the primary v2 calibrated judge lost to the strongest baseline.

    This is diagnostic only. It reads frozen v2 artifacts and does not recompute
    or mutate the v2 protocol, rankings, or decision gate.
    """
    results_dir = Path(results)
    out_dir = Path(output)
    _ensure_dir(out_dir)
    risk = pd.read_csv(results_dir / "v2_risk_coverage.csv")
    scenarios = pd.read_csv(results_dir / "v2_scenario_scores.csv")
    run_summary = _load_json(results_dir / "v2_run_summary.json")
    stats_path = results_dir.parent / "statistical_audit" / "v2_statistical_summary.json"
    stats = _load_json(stats_path) if stats_path.exists() else {}
    primary_judge = "calibration_selected_candidate_ranker"
    primary_risk_col = f"risk_{primary_judge}"
    primary_coverages = [0.05, 0.10]
    primary_rows = risk[(risk["judge_id"] == primary_judge) & (risk["coverage"].isin(primary_coverages))].copy()

    primary_vs_baseline = (
        primary_rows.groupby(["system_id", "badness_target"], as_index=False)
        .agg(
            mean_primary_far=("false_accept_rate", "mean"),
            mean_baseline_far=("baseline_far", "mean"),
            mean_absolute_margin=("absolute_margin", "mean"),
            mean_relative_margin=("relative_margin", "mean"),
            losing_row_rate=("absolute_margin", lambda s: float(np.mean(np.asarray(s) < 0.0))),
            row_count=("absolute_margin", "size"),
        )
        .sort_values(["system_id", "badness_target"])
    )
    _write_csv(out_dir / "primary_vs_baseline.csv", primary_vs_baseline)

    baseline_winner_counts = (
        primary_rows.groupby(["system_id", "badness_target", "baseline_judge"], as_index=False)
        .size()
        .rename(columns={"size": "winner_count"})
        .sort_values(["system_id", "badness_target", "winner_count"], ascending=[True, True, False])
    )
    _write_csv(out_dir / "baseline_winner_counts.csv", baseline_winner_counts)

    judge_far = (
        risk[risk["coverage"].isin(primary_coverages)]
        .groupby(["system_id", "badness_target", "judge_id"], as_index=False)
        .agg(mean_far=("false_accept_rate", "mean"))
        .sort_values(["system_id", "badness_target", "mean_far", "judge_id"])
    )
    judge_far["rank"] = judge_far.groupby(["system_id", "badness_target"])["mean_far"].rank(method="first")
    _write_csv(out_dir / "judge_far_ranking.csv", judge_far)

    risk_columns = [column for column in scenarios.columns if column.startswith("risk_")]
    auc_rows = []
    for (system_id, target), group in scenarios.groupby(["system_id", "badness_target"], sort=False):
        for column in risk_columns:
            auc_rows.append(
                {
                    "system_id": system_id,
                    "badness_target": target,
                    "judge_or_signal": column.replace("risk_", ""),
                    "auc": _safe_auc(group["bad_label"], group[column]),
                    "bad_rate": float(group["bad_label"].mean()),
                    "unique_scores": int(group[column].nunique()),
                }
            )
    auc_table = pd.DataFrame(auc_rows).sort_values(["system_id", "badness_target", "auc"], ascending=[True, True, False], na_position="last")
    _write_csv(out_dir / "judge_auc_by_target.csv", auc_table)

    label_balance = (
        scenarios.groupby(["system_id", "badness_target"], as_index=False)
        .agg(
            row_count=("bad_label", "size"),
            bad_rate=("bad_label", "mean"),
            rmse_mean=("rmse", "mean"),
            event_mismatch_rate=("event_mismatch", "mean"),
            primary_unique_scores=(primary_risk_col, "nunique"),
        )
        .sort_values(["system_id", "badness_target"])
    )
    _write_csv(out_dir / "label_balance.csv", label_balance)

    oracle_gap = (
        risk[(risk["coverage"].isin(primary_coverages)) & (risk["judge_id"].isin([primary_judge, "oracle_error_rank"]))]
        .pivot_table(
            index=["system_id", "badness_target", "coverage"],
            columns="judge_id",
            values="false_accept_rate",
            aggfunc="mean",
        )
        .reset_index()
    )
    if primary_judge in oracle_gap and "oracle_error_rank" in oracle_gap:
        oracle_gap["primary_minus_oracle_far"] = oracle_gap[primary_judge] - oracle_gap["oracle_error_rank"]
    _write_csv(out_dir / "oracle_gap_summary.csv", oracle_gap)

    moving_comparator = bool(
        baseline_winner_counts.groupby(["system_id", "badness_target"])["baseline_judge"].nunique().max() > 1
    )
    conformal_dominates = bool(
        baseline_winner_counts.groupby("baseline_judge")["winner_count"].sum().idxmax() == "conformal_risk_threshold"
    )
    event_failures = primary_vs_baseline[
        (primary_vs_baseline["badness_target"] == "bad_event")
        & (primary_vs_baseline["mean_absolute_margin"] < 0.0)
    ]["system_id"].tolist()
    no_signal_event_systems = label_balance[
        (label_balance["badness_target"] == "bad_event")
        & (label_balance["bad_rate"] <= 0.0)
    ]["system_id"].tolist()
    twotank_rmse = primary_vs_baseline[
        (primary_vs_baseline["system_id"] == "two_tank")
        & (primary_vs_baseline["badness_target"] == "bad_rmse")
    ]
    twotank_rmse_margin = float(twotank_rmse["mean_absolute_margin"].iloc[0]) if not twotank_rmse.empty else float("nan")

    findings = [
        {
            "finding": "The comparator is a moving strongest-baseline envelope.",
            "evidence": "baseline_judge varies across system/target/model/seed/coverage rows"
            if moving_comparator
            else "baseline_judge is stable",
            "interpretation": (
                "The primary margin is measured against the best observed baseline row-by-row, "
                "not against one fixed deployable baseline. This is conservative and explains "
                "small negative margins where the primary ties individual baselines on average."
            ),
        },
        {
            "finding": "conformal_risk_threshold is the dominant baseline winner.",
            "evidence": f"dominates winner counts: {conformal_dominates}",
            "interpretation": (
                "The strongest baseline is usually a single oriented signal threshold, which is simpler "
                "and less variable than the calibrated candidate family."
            ),
        },
        {
            "finding": "Event-risk is the clearest failure mode.",
            "evidence": f"negative event-risk systems: {event_failures}; degenerate event system(s): {no_signal_event_systems}",
            "interpretation": (
                "CSTR and heat_exchanger event-risk margins are negative, while TwoTank has no event positives. "
                "The calibrated candidate therefore has no robust event-risk support."
            ),
        },
        {
            "finding": "TwoTank RMSE underperformance is material.",
            "evidence": f"TwoTank bad_rmse mean absolute margin: {twotank_rmse_margin:.6f}",
            "interpretation": (
                "The primary calibrated candidate loses on the original TwoTank RMSE target after adding v2 "
                "models, targets, and stronger baselines."
            ),
        },
    ]
    summary = {
        "verdict": "UNDERPERFORMANCE_DIAGNOSED",
        "decision_gate": "NO_METHOD_CLAIM_BENCHMARK_ONLY",
        "primary_judge": primary_judge,
        "valid_systems": run_summary.get("valid_systems", []),
        "models_evaluated": run_summary.get("models_evaluated", []),
        "badness_targets": run_summary.get("badness_targets", []),
        "seed_count": run_summary.get("seed_count"),
        "statistical_verdict": stats.get("verdict"),
        "event_risk_worsening": stats.get("event_risk_worsening"),
        "positive_systems": stats.get("positive_systems"),
        "moving_baseline_comparator": moving_comparator,
        "dominant_baseline": baseline_winner_counts.groupby("baseline_judge")["winner_count"].sum().idxmax(),
        "findings": findings,
        "recommended_next_action": (
            "Do not attempt a claim upgrade. First separate fixed calibration-selected baseline comparison "
            "from row-wise strongest-baseline envelope comparison, then diagnose event-risk-specific signals."
        ),
    }
    _write_json(out_dir / "underperformance_diagnosis_summary.json", summary)

    top_judges = (
        judge_far.sort_values(["system_id", "badness_target", "mean_far"])
        .groupby(["system_id", "badness_target"])
        .head(5)
    )
    report = f"""# v2 Calibrated Underperformance Diagnosis

## Verdict

UNDERPERFORMANCE_DIAGNOSED

## Scope

This diagnosis reads existing frozen v2 artifacts only. It does not change the v2 protocol, does not rerun model training, and does not upgrade any claim.

## Primary Finding

The primary calibrated candidate underperforms because it is compared against a moving strongest-baseline envelope, and because event-risk targets expose failures that the calibrated ranker does not handle robustly.

## Primary vs Baseline

{_markdown_table(primary_vs_baseline, ["system_id", "badness_target", "mean_primary_far", "mean_baseline_far", "mean_absolute_margin", "losing_row_rate"])}

## Baseline Winner Counts

{_markdown_table(baseline_winner_counts.head(18), ["system_id", "badness_target", "baseline_judge", "winner_count"])}

## Best Judges by FAR

{_markdown_table(top_judges, ["system_id", "badness_target", "judge_id", "mean_far", "rank"], max_rows=30)}

## Label Balance

{_markdown_table(label_balance, ["system_id", "badness_target", "row_count", "bad_rate", "event_mismatch_rate", "primary_unique_scores"])}

## Root Causes

1. The current `baseline_far` is a row-wise envelope over baseline judges. That is stricter than comparing against one fixed deployable baseline and makes small negative margins likely even when the primary candidate ties a baseline on average.
2. `conformal_risk_threshold` is the dominant baseline winner, showing that simple oriented threshold rules often beat the calibrated candidate at low coverage.
3. Event-risk is the clearest failure mode: CSTR and heat_exchanger event targets have negative margins, while TwoTank event labels are degenerate.
4. The primary candidate is not the best calibrated-family member on TwoTank RMSE; learned/logistic scores rank that target better in the current artifacts.

## Claim Impact

The v2 decision remains `NO_METHOD_CLAIM_BENCHMARK_ONLY`. This diagnosis does not support a claim upgrade.

## Recommended Next Action

First produce a calibration-selected fixed-baseline comparison beside the current row-wise strongest-baseline envelope. Then run an event-risk-specific failure analysis before changing any judge.
"""
    Path("reports/v2_calibrated_underperformance_diagnosis.md").write_text(report, encoding="utf-8")
    return summary
