from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from scs.experiments.registry import make_system
from scs.experiments.v2 import (
    _make_v2_model,
    _markdown_table,
    _risk_curve_from_labels,
    generate_v2_dataset,
    load_event_config,
    load_v2_config,
    trajectory_bad_event,
)
from scs.metrics.trajectory import mae, max_abs_error, rmse
from scs.models.user_model import (
    load_user_model_from_spec,
    validate_rollout_shape,
    validate_user_model_interface,
)
from scs.validators.invariants import invariant_residual_score
from scs.validators.repair import repair_amount_score
from scs.validators.support import SupportDistance


PACKAGE_ID = "v2_public_benchmark_hardening"
PUBLIC_HOOK = "A benchmark for testing whether learned dynamical simulators know when to refuse counterfactual predictions."
ALLOWED_CLAIM = "benchmark exposes target-dependent calibrated-refusal failure"
PUBLIC_FINDING = "calibrated refusal is target-dependent and not reliable for event-risk"
PRODUCT_DIRS = {"api", "frontend", "dashboard", "database", "auth"}
OLD_REPO_NAMES = [
    "time" + "-series" + "-simulator",
    "digital" + "-twin" + "-engine",
    "flux" + "-attention" + "-engine",
    "plant" + "-scenario" + "-compiler",
]
BENCHMARK_SYSTEMS = ["cstr", "heat_exchanger"]
PUBLIC_JUDGES = ["support_only", "invariant_only", "repair_only", "event_guard_public"]
BADNESS_TARGETS = ["bad_rmse", "bad_event", "bad_rmse_or_event"]
PUBLIC_COVERAGES = [0.05, 0.10, 0.20, 0.40, 0.80, 1.00]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    _ensure_dir(target.parent)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_csv(path: str | Path, frame: pd.DataFrame) -> None:
    target = Path(path)
    _ensure_dir(target.parent)
    frame.to_csv(target, index=False)


def _read_yaml(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return data


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_output(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:  # pragma: no cover - local git state dependent
        return f"unknown: {exc}"


def load_public_hardening_config(path: str | Path) -> dict[str, Any]:
    config = _read_yaml(path)
    required = {
        "package_id",
        "source_branch",
        "source_commit",
        "controlling_reports",
        "source_artifacts",
        "public_claim",
        "public_hook",
        "forbidden_claims",
        "benchmark_api",
        "public_outputs",
        "forbidden",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"missing public hardening config keys: {missing}")
    if config["package_id"] != PACKAGE_ID:
        raise ValueError(f"unexpected package_id: {config['package_id']}")
    if config["public_hook"]["text"] != PUBLIC_HOOK:
        raise ValueError("public hook changed")
    forbidden = config["forbidden"]
    for key in [
        "allow_new_experiments",
        "allow_new_systems",
        "allow_new_models",
        "allow_new_judges",
        "allow_new_signals",
        "allow_protocol_mutation",
        "allow_prior_artifact_overwrite",
    ]:
        if forbidden.get(key) is not False:
            raise ValueError(f"config must forbid {key}")
    return config


def _source_paths(config: dict[str, Any]) -> dict[str, Path]:
    paths = {name: Path(path) for name, path in config["source_artifacts"].items()}
    paths.update({name: Path(path) for name, path in config["controlling_reports"].items()})
    return paths


def _scan_forbidden_runtime_refs() -> dict[str, list[str]]:
    old_repo_hits: list[str] = []
    path_hack_hits: list[str] = []
    for root in [Path("src"), Path("scripts"), Path("tests"), Path("examples")]:
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
                if ("PYTHON" + "PATH") in stripped:
                    path_hack_hits.append(str(path))
    return {
        "old_repo_runtime_import_hits": sorted(set(old_repo_hits)),
        "path_hack_hits": sorted(set(path_hack_hits)),
    }


def scan_public_claim_language(text: str, forbidden_claims: list[str]) -> list[str]:
    """Return forbidden claim hits outside explicit non-claim/forbidden contexts."""
    hits: list[str] = []
    context = "body"
    context_markers = [
        "forbidden",
        "non-goal",
        "non-goals",
        "non-intended",
        "non-claim",
        "non-claims",
        "not claim",
        "claim boundaries",
        "not supported",
        "what this benchmark does not",
        "what it does not",
    ]
    negated_line_markers = [
        "does not claim",
        "does not support",
        "does not provide",
        "does not test",
        "do not use",
        "is not a",
        "is not an",
        "is not the",
        "is not ",
        "not a robust",
        "not a method-success claim",
        "not a product",
    ]
    for raw_line in text.splitlines():
        line = raw_line.strip()
        lower = line.lower()
        if lower.startswith("## ") or lower.startswith("# "):
            if any(token in lower for token in context_markers):
                context = "forbidden"
            else:
                context = "body"
        marker_line = lower.strip(" *")
        if marker_line.endswith(":") and any(token in marker_line for token in context_markers):
            context = "forbidden"
            continue
        if context == "forbidden":
            continue
        if any(token in lower for token in negated_line_markers):
            continue
        for claim in forbidden_claims:
            if claim.lower() in lower:
                hits.append(claim)
    return sorted(set(hits))


def verify_public_benchmark_hardening_preconditions(config_path: str | Path, output: str | Path) -> dict[str, Any]:
    config = load_public_hardening_config(config_path)
    out_dir = Path(output)
    _ensure_dir(out_dir)
    paths = _source_paths(config)
    missing = [str(path) for path in paths.values() if not path.exists() or path.stat().st_size == 0]
    hashes = {name: {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size} for name, path in paths.items() if path.exists()}
    scientific_gate = Path(config["controlling_reports"]["v2_scientific_gate"]).read_text(encoding="utf-8")
    comparator_gate = Path(config["controlling_reports"]["comparator_fairness_gate"]).read_text(encoding="utf-8")
    taxonomy = Path(config["controlling_reports"]["comparator_taxonomy"]).read_text(encoding="utf-8")
    status = _git_output(["status", "--short"])
    status_paths = {line[3:].strip() for line in status.splitlines() if len(line) >= 4}
    source_paths = {str(path) for path in paths.values()}
    product_dirs = sorted(str(path) for path in Path(".").iterdir() if path.is_dir() and path.name in PRODUCT_DIRS)
    dependency_scan = _scan_forbidden_runtime_refs()
    reasons: list[str] = []
    if missing:
        reasons.append(f"missing source artifacts: {missing}")
    if "NO_METHOD_CLAIM_BENCHMARK_ONLY" not in scientific_gate:
        reasons.append("v2 scientific gate is not NO_METHOD_CLAIM_BENCHMARK_ONLY")
    if "CALIBRATED_TARGET_DEPENDENT" not in comparator_gate:
        reasons.append("comparator fairness gate is not CALIBRATED_TARGET_DEPENDENT")
    if "row-wise envelope is diagnostic only" not in taxonomy.lower() and "diagnostic-only comparator" not in taxonomy.lower():
        reasons.append("row-wise envelope diagnostic-only text missing")
    if product_dirs:
        reasons.append(f"product/API/frontend directories detected: {product_dirs}")
    if dependency_scan["old_repo_runtime_import_hits"] or dependency_scan["path_hack_hits"]:
        reasons.append("forbidden runtime refs or path hacks detected")
    source_artifact_modified = bool(status_paths.intersection(source_paths))
    if source_artifact_modified:
        reasons.append("prior source artifacts are modified")
    verdict = "READY_FOR_PUBLIC_BENCHMARK_HARDENING" if not reasons else "NOT_READY"
    result = {
        "package_id": PACKAGE_ID,
        "working_tree_status": status,
        "source_artifacts": {name: str(path) for name, path in paths.items()},
        "missing_source_artifacts": missing,
        "v2_scientific_gate_ok": "NO_METHOD_CLAIM_BENCHMARK_ONLY" in scientific_gate,
        "comparator_fairness_gate_ok": "CALIBRATED_TARGET_DEPENDENT" in comparator_gate,
        "row_wise_envelope_diagnostic_only": "diagnostic" in taxonomy.lower() and "row-wise" in taxonomy.lower(),
        "product_dirs": product_dirs,
        "forbidden_dependency_scan": dependency_scan,
        "source_artifact_modified": source_artifact_modified,
        "verdict": verdict,
        "reasons": reasons,
    }
    _write_json(out_dir / "source_artifact_hashes.json", hashes)
    _write_json(out_dir / "precondition_check.json", result)
    report = f"""# v2 Public Benchmark Hardening Preconditions

## Verdict

{verdict}

## Source Artifacts

Missing: {missing or ["none"]}

Hashes written: {len(hashes)}

## Gate Checks

- v2 scientific gate benchmark-only: {result["v2_scientific_gate_ok"]}
- comparator fairness target-dependent: {result["comparator_fairness_gate_ok"]}
- row-wise envelope diagnostic only: {result["row_wise_envelope_diagnostic_only"]}

## Integrity Checks

- prior source artifact modified: {source_artifact_modified}
- product directories: {product_dirs}
- dependency scan: {dependency_scan}

## Reasons

{reasons or ["none"]}
"""
    Path("reports/v2_public_benchmark_hardening_preconditions.md").write_text(report, encoding="utf-8")
    return result


def _percentile_rank(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(reference, dtype=float))
    return np.searchsorted(reference, np.asarray(values, dtype=float), side="right") / max(len(reference), 1)


def _event_guard(calibration: pd.DataFrame, table: pd.DataFrame) -> np.ndarray:
    cols = ["risk_invariant_only", "risk_disagreement_only", "risk_support_only"]
    ranks = [_percentile_rank(calibration[col].to_numpy(dtype=float), table[col].to_numpy(dtype=float)) for col in cols]
    return np.maximum.reduce(ranks)


def _make_public_model(model_id: str, seed: int) -> object:
    config = load_v2_config("configs/v2/v2_scientific_strengthening.yaml")
    return _make_v2_model(model_id, seed, config)


def _model_specs_from_args(custom_model: str | None, models: list[str] | None) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    if custom_model:
        model = load_user_model_from_spec(custom_model)
        specs.append({"model": model, "model_id": model.model_id, "is_custom_model": True, "is_builtin_model": False})
    for model_id in models or []:
        model = _make_public_model(model_id, seed=13)
        specs.append({"model": model, "model_id": model_id, "is_custom_model": False, "is_builtin_model": True})
    if not specs:
        raise ValueError("provide --model file.py:ClassName or --models built_in_model ...")
    return specs


def _target_labels(frame: pd.DataFrame, target: str) -> np.ndarray:
    if target == "bad_rmse":
        return frame["rmse"].to_numpy(dtype=float) > 0.20
    if target == "bad_event":
        return frame["event_bad"].to_numpy(dtype=bool)
    if target == "bad_rmse_or_event":
        return (frame["rmse"].to_numpy(dtype=float) > 0.20) | frame["event_bad"].to_numpy(dtype=bool)
    raise ValueError(f"unknown target: {target}")


def _risk_scores(frame: pd.DataFrame, judge_id: str) -> np.ndarray:
    return frame[f"risk_{judge_id}"].to_numpy(dtype=float)


def run_public_benchmark(
    output: str | Path,
    custom_model: str | None = None,
    models: list[str] | None = None,
    command: str | None = None,
) -> dict[str, Any]:
    out_dir = Path(output)
    _ensure_dir(out_dir)
    v2_config = load_v2_config("configs/v2/v2_scientific_strengthening.yaml")
    event_config = load_event_config("configs/v2/v2_event_targets.yaml")
    model_specs = _model_specs_from_args(custom_model, models)
    scenario_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for system_id in BENCHMARK_SYSTEMS:
        dataset = generate_v2_dataset(system_id, v2_config, seed=21)
        system = make_system(system_id)
        support = SupportDistance()
        support.fit(dataset["model_train"])
        for spec in model_specs:
            model = spec["model"]
            if spec["is_builtin_model"]:
                model = _make_public_model(str(spec["model_id"]), seed=21)
            validate_user_model_interface(model)
            model.fit(dataset["model_train"])
            rows: list[dict[str, Any]] = []
            public_splits = [
                split
                for split in dataset
                if split.startswith("judge_calibration") or split.startswith("judge_test")
            ]
            for split in public_splits:
                batch = dataset[split]
                for idx in range(batch.n_trajectories):
                    pred = model.predict_rollout(batch.states[idx, 0], batch.actions[idx], batch.disturbances[idx])
                    pred = validate_rollout_shape(pred, batch.states[idx, 0], batch.actions[idx])
                    truth = batch.states[idx]
                    inv = invariant_residual_score(system, pred, batch.actions[idx], batch.disturbances[idx], float(v2_config["data"]["dt"]))
                    repair = repair_amount_score(system, pred)
                    supp = support.score(batch.actions[idx], batch.disturbances[idx])
                    pred_event = trajectory_bad_event(system_id, pred, event_config)
                    true_event = trajectory_bad_event(system_id, truth, event_config)
                    rows.append(
                        {
                            "system_id": system_id,
                            "split": split,
                            "role": "calibration" if split.startswith("judge_calibration") else "test",
                            "scenario_id": f"{system_id}_{split}_{idx:04d}",
                            "scenario_type": batch.scenario_type[idx],
                            "model_id": str(spec["model_id"]),
                            "is_custom_model": bool(spec["is_custom_model"]),
                            "is_builtin_model": bool(spec["is_builtin_model"]),
                            "rmse": rmse(pred, truth),
                            "mae": mae(pred, truth),
                            "max_abs_error": max_abs_error(pred, truth),
                            "true_event": bool(true_event),
                            "pred_event": bool(pred_event),
                            "event_bad": bool(true_event != pred_event),
                            "risk_support_only": supp,
                            "risk_invariant_only": inv,
                            "risk_repair_only": repair,
                            "risk_disagreement_only": 0.0,
                        }
                    )
            frame = pd.DataFrame(rows)
            cal = frame[frame["role"] == "calibration"].copy()
            test = frame[frame["role"] == "test"].copy()
            frame.loc[frame["role"] == "calibration", "risk_event_guard_public"] = _event_guard(cal, cal)
            frame.loc[frame["role"] == "test", "risk_event_guard_public"] = _event_guard(cal, test)
            scenario_frames.append(frame)
            test = frame[frame["role"] == "test"].copy()
            metric_rows.append(
                {
                    "system_id": system_id,
                    "model_id": str(spec["model_id"]),
                    "is_custom_model": bool(spec["is_custom_model"]),
                    "is_builtin_model": bool(spec["is_builtin_model"]),
                    "rmse_mean": float(test["rmse"].mean()),
                    "mae_mean": float(test["mae"].mean()),
                    "max_abs_error_mean": float(test["max_abs_error"].mean()),
                    "n_scenarios": int(len(test)),
                }
            )
            event_rows.append(
                {
                    "system_id": system_id,
                    "model_id": str(spec["model_id"]),
                    "true_event_rate": float(test["true_event"].mean()),
                    "pred_event_rate": float(test["pred_event"].mean()),
                    "event_mismatch_rate": float(test["event_bad"].mean()),
                    "n_scenarios": int(len(test)),
                }
            )
    scenarios = pd.concat(scenario_frames, ignore_index=True)
    test_scenarios = scenarios[scenarios["role"] == "test"].copy()
    risk_rows: list[pd.DataFrame] = []
    accepted_rows: list[dict[str, Any]] = []
    for (system_id, model_id), group in test_scenarios.groupby(["system_id", "model_id"], sort=True):
        for target in BADNESS_TARGETS:
            labels = _target_labels(group, target)
            for judge_id in PUBLIC_JUDGES:
                scores = _risk_scores(group, judge_id)
                curve = _risk_curve_from_labels(labels, scores, PUBLIC_COVERAGES)
                curve["system_id"] = system_id
                curve["model_id"] = model_id
                curve["badness_target"] = target
                curve["judge_id"] = judge_id
                curve["is_custom_model"] = bool(group["is_custom_model"].iloc[0])
                curve["is_builtin_model"] = bool(group["is_builtin_model"].iloc[0])
                risk_rows.append(curve)
                for coverage in [0.05, 0.10]:
                    n = min(max(int(math.ceil(coverage * len(group))), 1), len(group))
                    order = np.argsort(scores, kind="mergesort")
                    accepted_idx = set(order[:n])
                    for pos, (_, row) in enumerate(group.iterrows()):
                        accepted = pos in accepted_idx
                        false_accept = bool(accepted and labels[pos])
                        if false_accept or (target == "bad_event" and judge_id == "event_guard_public" and accepted):
                            accepted_rows.append(
                                {
                                    "system_id": system_id,
                                    "model_id": model_id,
                                    "scenario_id": row["scenario_id"],
                                    "scenario_type": row["scenario_type"],
                                    "badness_target": target,
                                    "judge_id": judge_id,
                                    "coverage": coverage,
                                    "rmse": float(row["rmse"]),
                                    "event_bad": bool(row["event_bad"]),
                                    "risk_score": float(scores[pos]),
                                    "accepted": bool(accepted),
                                    "false_accept": false_accept,
                                }
                            )
    risk_coverage = pd.concat(risk_rows, ignore_index=True)
    model_metrics = pd.DataFrame(metric_rows)
    event_metrics = pd.DataFrame(event_rows)
    accepted_false_accepts = pd.DataFrame(accepted_rows)
    if accepted_false_accepts.empty:
        accepted_false_accepts = pd.DataFrame(
            columns=[
                "system_id",
                "model_id",
                "scenario_id",
                "scenario_type",
                "badness_target",
                "judge_id",
                "coverage",
                "rmse",
                "event_bad",
                "risk_score",
                "accepted",
                "false_accept",
            ]
        )
    for name, frame in [
        ("risk_coverage.csv", risk_coverage),
        ("model_metrics.csv", model_metrics),
        ("event_metrics.csv", event_metrics),
        ("accepted_false_accepts.csv", accepted_false_accepts),
    ]:
        if frame.isna().any().any():
            raise RuntimeError(f"{name} contains NaN")
        _write_csv(out_dir / name, frame)
    summary = {
        "verdict": "PUBLIC_BENCHMARK_RUN_COMPLETE",
        "command": command or "",
        "is_current_scientific_evidence": False,
        "models": sorted(risk_coverage["model_id"].unique().tolist()),
        "systems": BENCHMARK_SYSTEMS,
        "badness_targets": BADNESS_TARGETS,
        "risk_rows": int(len(risk_coverage)),
        "accepted_false_accept_rows": int(len(accepted_false_accepts)),
        "event_risk_present": "bad_event" in set(risk_coverage["badness_target"]),
        "current_claim_updated": False,
    }
    _write_json(out_dir / "benchmark_summary.json", summary)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    plot = risk_coverage.groupby(["badness_target", "coverage"], as_index=False)["false_accept_rate"].mean()
    for target, group in plot.groupby("badness_target"):
        ax.plot(group["coverage"], group["false_accept_rate"], marker="o", label=target)
    ax.set_xlabel("coverage")
    ax.set_ylabel("false accept rate")
    ax.set_title("Public benchmark risk-coverage")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "risk_coverage.png", dpi=160)
    plt.close(fig)
    report = f"""# Public Benchmark Run Report

## Command

```bash
{command or "python scripts/run_benchmark.py"}
```

## Models evaluated

{", ".join(summary["models"])}

## Systems evaluated

{", ".join(BENCHMARK_SYSTEMS)}

## Badness targets

{", ".join(BADNESS_TARGETS)}

## Risk-coverage summary

{_markdown_table(risk_coverage.groupby(["badness_target", "judge_id"], as_index=False)["false_accept_rate"].mean(), ["badness_target", "judge_id", "false_accept_rate"], max_rows=16)}

## Event-risk summary

{_markdown_table(event_metrics, ["system_id", "model_id", "event_mismatch_rate", "n_scenarios"], max_rows=12)}

## Accepted false accepts

Rows written: {len(accepted_false_accepts)}

## What this run proves

This run proves that the selected model can be evaluated through the public benchmark interface and produces risk-coverage artifacts.

## What this run does not prove

This benchmark run does not update the repository's current scientific claim. It does not prove simulator trustworthiness, safety certification, product readiness, or general reliability.
"""
    (out_dir / "benchmark_report.md").write_text(report, encoding="utf-8")
    return summary


def build_public_event_risk_figure(config_path: str | Path, output: str | Path) -> dict[str, Any]:
    config = load_public_hardening_config(config_path)
    stats = _read_json(config["source_artifacts"]["comparator_stats"])
    rmse = float(stats["rmse_target_result"]["mean_margin"])
    event = float(stats["event_target_result"]["mean_margin"])
    out = Path(output)
    _ensure_dir(out.parent)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#4C78A8", "#E45756"]
    ax.bar(["RMSE target", "Event target"], [rmse, event], color=colors)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_ylabel("Mean FAR margin vs fair baseline")
    ax.set_title("RMSE can look near-neutral while event-risk worsens")
    ax.text(
        0.5,
        -0.22,
        "v2 calibrated refusal is target-dependent; this is benchmark evidence, not a method-success claim.",
        ha="center",
        transform=ax.transAxes,
        fontsize=9,
    )
    for i, value in enumerate([rmse, event]):
        ax.text(i, value, f"{value:+.5f}", ha="center", va="bottom" if value >= 0 else "top")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    manifest = {
        "verdict": "PUBLIC_EVENT_RISK_FIGURE_BUILT",
        "output": str(out),
        "source": config["source_artifacts"]["comparator_stats"],
        "rmse_target_mean_margin": rmse,
        "event_target_mean_margin": event,
        "caption": "v2 calibrated refusal is target-dependent; this is benchmark evidence, not a method-success claim.",
    }
    _write_json("results/v2_public_benchmark_hardening/figures/event_risk_figure_manifest.json", manifest)
    report = f"""# v2 Public Event-Risk Figure

## Verdict

PUBLIC_EVENT_RISK_FIGURE_BUILT

## Source Values

- RMSE target mean margin: {rmse:+.6f}
- Event target mean margin: {event:+.6f}

## Claim Boundary

The figure shows target dependence and does not claim method success.
"""
    Path("reports/v2_public_event_risk_figure.md").write_text(report, encoding="utf-8")
    return manifest


def _example_from_row(row: pd.Series, title: str, source_index: int, accepted: bool, false_accept: bool) -> dict[str, Any]:
    return {
        "title": title,
        "system_id": str(row["system_id"]),
        "model_id": str(row["model_id"]),
        "scenario_id": str(row.get("scenario_id", f"source_row_{source_index}")),
        "scenario_type": str(row.get("scenario_type", "unknown")),
        "badness_target": str(row.get("badness_target", "bad_event")),
        "judge_id": str(row.get("judge_id", row.get("calibrated_judge_id", "calibration_selected_candidate_ranker"))),
        "coverage": float(row.get("coverage", 0.05)),
        "accepted": bool(accepted),
        "false_accept": bool(false_accept),
        "rmse": float(row.get("rmse", row.get("calibrated_far", 0.0))),
        "event_bad": bool(row.get("event_bad", false_accept)),
        "risk_score": float(row.get("risk_score", row.get("calibrated_far", 0.0))),
        "baseline_judge": str(row.get("baseline_judge_id", row.get("fair_baseline_judge_id", ""))),
        "source_row_id": int(source_index),
        "trajectory_note": "Trajectory plot unavailable from current artifacts.",
    }


def build_public_failure_gallery(config_path: str | Path, output: str | Path, figure_dir: str | Path) -> dict[str, Any]:
    config = load_public_hardening_config(config_path)
    scenario = pd.read_csv(config["source_artifacts"]["v2_scenario_scores"])
    comparator = pd.read_csv(config["source_artifacts"]["comparator_fairness"])
    event = scenario[scenario["badness_target"] == "bad_event"].copy()
    primary = event[event["risk_calibration_selected_candidate_ranker"].notna()].copy()
    primary = primary.reset_index().rename(columns={"index": "source_index"})
    primary["coverage"] = 0.05
    primary["judge_id"] = "calibration_selected_candidate_ranker"
    primary["risk_score"] = primary["risk_calibration_selected_candidate_ranker"].astype(float)
    primary["accepted"] = False
    for _, group in primary.groupby(["system_id", "model_id"], sort=False):
        n_accept = min(max(int(math.ceil(0.05 * len(group))), 1), len(group))
        accepted_index = group.sort_values("risk_score", kind="mergesort").head(n_accept).index
        primary.loc[accepted_index, "accepted"] = True
    examples: list[dict[str, Any]] = []
    false_accept = primary[(primary["accepted"]) & (primary["bad_label"].astype(bool)) & (primary["event_mismatch"].astype(bool))].head(1)
    if not false_accept.empty:
        row = false_accept.iloc[0]
        examples.append(
            _example_from_row(
                row.rename({"event_mismatch": "event_bad"}),
                "accepted event-risk false accept",
                int(row["source_index"]),
                True,
                True,
            )
        )
    rmse_ok_event_bad = primary[(primary["rmse"] <= 0.20) & (primary["event_mismatch"].astype(bool))].head(1)
    if not rmse_ok_event_bad.empty:
        row = rmse_ok_event_bad.iloc[0]
        examples.append(
            _example_from_row(
                row.rename({"event_mismatch": "event_bad"}),
                "RMSE-acceptable but event-bad case",
                int(row["source_index"]),
                bool(row["accepted"]),
                bool(row["accepted"]),
            )
        )
    loss = comparator[
        (comparator["badness_target"] == "bad_event")
        & (comparator["comparator_mode"] == "per_system_target_calibration_selected_baseline")
        & (comparator["absolute_margin"] < 0.0)
    ].reset_index().head(1)
    if not loss.empty:
        row = loss.iloc[0]
        examples.append(_example_from_row(row, "calibrated judge loses to fair deployable baseline", int(row["index"]), True, True))
    envelope = comparator[
        (comparator["badness_target"] == "bad_event")
        & (comparator["comparator_mode"] == "row_wise_strongest_baseline_envelope")
        & (comparator["absolute_margin"] < 0.0)
    ].reset_index().head(1)
    if not envelope.empty:
        row = envelope.iloc[0]
        examples.append(_example_from_row(row, "row-wise envelope illustrates diagnostic ceiling", int(row["index"]), True, True))
    refused = primary[(~primary["accepted"]) & (primary["bad_label"].astype(bool)) & (primary["event_mismatch"].astype(bool))].head(1)
    if not refused.empty:
        row = refused.iloc[0]
        examples.append(
            _example_from_row(
                row.rename({"event_mismatch": "event_bad"}),
                "correctly refused event-risk case",
                int(row["source_index"]),
                False,
                False,
            )
        )
    if len(examples) < 5:
        raise RuntimeError("failure gallery requires at least 5 examples")
    fig_dir = Path(figure_dir)
    _ensure_dir(fig_dir)
    out = Path(output)
    _ensure_dir(out.parent)
    lines = ["# Event-Risk Failure Gallery", ""]
    for idx, example in enumerate(examples[:5], start=1):
        lines.extend(
            [
                f"## Example {idx}: {example['title']}",
                "",
                f"- system_id: {example['system_id']}",
                f"- model_id: {example['model_id']}",
                f"- scenario_id: {example['scenario_id']}",
                f"- scenario_type: {example['scenario_type']}",
                f"- badness_target: {example['badness_target']}",
                f"- judge_id: {example['judge_id']}",
                f"- coverage: {example['coverage']}",
                f"- accepted/refused: {'accepted' if example['accepted'] else 'refused'}",
                f"- false_accept: {example['false_accept']}",
                f"- rmse: {example['rmse']:.6f}",
                f"- event_bad: {example['event_bad']}",
                f"- risk_score: {example['risk_score']:.6f}",
                f"- baseline_judge: {example['baseline_judge']}",
                f"- source_row_id: {example['source_row_id']}",
                f"- {example['trajectory_note']}",
                "",
            ]
        )
    out.write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "verdict": "PUBLIC_FAILURE_GALLERY_BUILT",
        "example_count": len(examples[:5]),
        "output": str(out),
        "figure_dir": str(fig_dir),
        "examples": examples[:5],
        "trajectory_plots_fabricated": False,
    }
    _write_json("results/v2_public_benchmark_hardening/failure_gallery_manifest.json", manifest)
    report = f"""# v2 Public Failure Gallery Report

## Verdict

PUBLIC_FAILURE_GALLERY_BUILT

## Examples

{len(examples[:5])}

## Source

Examples are selected from frozen v2 scenario scores and comparator fairness rows. Trajectory plot unavailable from current artifacts.
"""
    Path("reports/v2_public_failure_gallery_report.md").write_text(report, encoding="utf-8")
    return manifest


def build_public_benchmark_manifest(config_path: str | Path, output: str | Path) -> dict[str, Any]:
    config = load_public_hardening_config(config_path)
    source_hashes = {name: _sha256(path) for name, path in _source_paths(config).items() if path.exists()}
    manifest = {
        "package_id": PACKAGE_ID,
        "release_type": "public_benchmark_prototype",
        "allowed_claim": ALLOWED_CLAIM,
        "method_claim_supported": False,
        "included_docs": [
            "README.md",
            "docs/v2/benchmark_api_contract.md",
            "docs/v2/benchmark_task.md",
            "docs/v2/benchmark_card.md",
            "docs/v2/event_risk_failure_gallery.md",
        ],
        "included_scripts": [
            "scripts/run_benchmark.py",
            "scripts/v2_build_public_event_risk_figure.py",
            "scripts/v2_build_public_failure_gallery.py",
            "scripts/v2_check_public_benchmark_package.py",
        ],
        "included_figures": ["docs/v2/figures/event_risk_vs_rmse_public.png"],
        "source_artifact_hashes": source_hashes,
        "forbidden_claims": list(config["forbidden_claims"]),
    }
    _write_json(output, manifest)
    return manifest


def check_public_benchmark_package(config_path: str | Path, manifest_path: str | Path) -> dict[str, Any]:
    config = load_public_hardening_config(config_path)
    manifest = _read_json(manifest_path)
    reasons: list[str] = []
    if manifest.get("method_claim_supported") is not False:
        reasons.append("manifest supports method claim")
    custom = run_public_benchmark("results/public_benchmark_run", custom_model="examples/custom_model_example.py:DampedLinearUserModel", command=config["benchmark_api"]["command"])
    builtin = run_public_benchmark("results/public_benchmark_builtin", models=["linear_narx", "mlp_state_space"], command="python scripts/run_benchmark.py --models linear_narx mlp_state_space --output results/public_benchmark_builtin")
    required_paths = [
        "docs/v2/benchmark_api_contract.md",
        "docs/v2/benchmark_task.md",
        "docs/v2/benchmark_card.md",
        "docs/v2/event_risk_failure_gallery.md",
        "docs/v2/figures/event_risk_vs_rmse_public.png",
        "README.md",
    ]
    missing = [path for path in required_paths if not Path(path).exists() or Path(path).stat().st_size == 0]
    if missing:
        reasons.append(f"missing package files: {missing}")
    readme = Path("README.md").read_text(encoding="utf-8")
    if not readme.startswith("# Selective Counterfactual Simulation Benchmark"):
        reasons.append("README does not start with public benchmark title")
    text_paths = [path for path in required_paths if Path(path).suffix.lower() in {".md", ""}]
    docs_text = "\n".join(Path(path).read_text(encoding="utf-8") for path in text_paths if Path(path).exists() and Path(path).is_file())
    claim_hits = scan_public_claim_language(docs_text, list(config["forbidden_claims"]))
    if claim_hits:
        reasons.append(f"forbidden claim language: {claim_hits}")
    for name, expected_hash in manifest.get("source_artifact_hashes", {}).items():
        path = _source_paths(config).get(name)
        if path is None or not path.exists() or _sha256(path) != expected_hash:
            reasons.append(f"source artifact hash changed: {name}")
    if custom.get("verdict") != "PUBLIC_BENCHMARK_RUN_COMPLETE":
        reasons.append("custom model benchmark failed")
    if builtin.get("verdict") != "PUBLIC_BENCHMARK_RUN_COMPLETE":
        reasons.append("built-in benchmark failed")
    verdict = "V2_PUBLIC_BENCHMARK_PACKAGE_ACCEPTED" if not reasons else "V2_PUBLIC_BENCHMARK_PACKAGE_REJECTED"
    result = {
        "verdict": verdict,
        "custom_model_benchmark_executed": True,
        "builtin_benchmark_executed": True,
        "event_risk_figure_exists": Path("docs/v2/figures/event_risk_vs_rmse_public.png").exists(),
        "claim_language_hits": claim_hits,
        "source_hashes_checked": len(manifest.get("source_artifact_hashes", {})),
        "method_claim_supported": False,
        "reasons": reasons,
    }
    _write_json("results/v2_public_benchmark_hardening/package_check.json", result)
    report = f"""# v2 Public Benchmark Package Check

## Verdict

{verdict}

## Benchmark Commands

- custom model benchmark executed: {result["custom_model_benchmark_executed"]}
- built-in benchmark executed: {result["builtin_benchmark_executed"]}

## Claim Language

Forbidden hits: {claim_hits or ["none"]}

## Source Hashes

Checked: {result["source_hashes_checked"]}

## Reasons

{reasons or ["none"]}
"""
    Path("reports/v2_public_benchmark_package_check.md").write_text(report, encoding="utf-8")
    if verdict != "V2_PUBLIC_BENCHMARK_PACKAGE_ACCEPTED":
        raise RuntimeError(f"public benchmark package rejected: {reasons}")
    return result
