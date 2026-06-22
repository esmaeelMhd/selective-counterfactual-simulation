from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from scs.data.generate import generate_dataset
from scs.experiments.registry import make_model, make_system
from scs.metrics.selective import risk_coverage_curve
from scs.metrics.trajectory import final_state_error, mae, max_abs_error, rmse
from scs.models.base import SimulatorModel
from scs.systems.base import TrajectoryBatch
from scs.validators.disagreement import disagreement_score
from scs.validators.invariants import invariant_residual_score
from scs.validators.judges import compute_judge_score_frame
from scs.validators.repair import repair_amount_score
from scs.validators.support import SupportDistance
from scs.validators.uncertainty import uncertainty_score
from scs.experiments.current_status import README_END, README_START, _git_dirty_lines, _markdown_table, _scan_forbidden_runtime_refs
from scs.experiments.technical_note_package import scan_forbidden_claim_language


USABILITY_START = "<!-- SCS_USABILITY_START -->"
USABILITY_END = "<!-- SCS_USABILITY_END -->"
USABILITY_RESULTS_ROOT = Path("results/benchmark_usability")
ALLOWED_CLAIM = "A weak but positive low-coverage refusal benchmark under a frozen protocol."
PROTECTED_ARTIFACTS = [
    "reports/current_status_decision_gate.md",
    "reports/practical_utility_decision_gate.md",
    "reports/repair_signal_role_decision_gate.md",
    "results/current_status/evidence_manifest/current_evidence_manifest.json",
]
REQUIRED_README_SECTIONS = [
    "## Quickstart",
    "## Run the Current Status Demo",
    "## What This Benchmark Tests",
    "## What This Benchmark Does Not Test",
    "## Add Your Own Model",
    "## Local Model Comparison",
    "## Current Evidence Status",
    "## Reproducibility",
    "## Claim Boundaries",
]
FORBIDDEN_SURFACE_DIRS = ["api", "frontend", "dashboard", "web", "database"]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    _ensure_dir(target.parent)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_yaml(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return data


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _gate_json(markdown_path: str | Path) -> dict[str, Any]:
    json_path = Path(markdown_path).with_suffix(".json")
    if json_path.exists():
        return _read_json(json_path)
    return {"text": Path(markdown_path).read_text(encoding="utf-8")}


def _format_command(args: list[str]) -> str:
    return " ".join(args)


def load_usability_config(path: str | Path) -> dict[str, Any]:
    config = _read_yaml(path)
    required = {
        "package_id",
        "source_commit",
        "source_tag",
        "controlling_status",
        "allowed_claim",
        "expansion_policy",
        "allowed_usability_additions",
        "forbidden_claims",
        "forbidden_research_expansion",
        "demo",
        "custom_model",
        "model_comparison",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"missing usability config keys: {missing}")
    if config["package_id"] != "benchmark_usability_v1_1":
        raise ValueError("unexpected package_id")
    if config["allowed_claim"]["text"] != ALLOWED_CLAIM:
        raise ValueError("allowed claim changed")
    policy = config["expansion_policy"]
    if policy.get("scientific_expansion_allowed") is not False:
        raise ValueError("scientific expansion must be false")
    if policy.get("usability_expansion_allowed") is not True:
        raise ValueError("usability expansion must be true")
    serialized_forbidden = yaml.safe_dump(config["forbidden_research_expansion"]).lower()
    for required_token in ["rssm", "heat_exchanger", "third system", "product/api/frontend"]:
        if required_token not in serialized_forbidden:
            raise ValueError(f"forbidden research expansion must include {required_token}")
    return config


def verify_benchmark_usability_preconditions(
    config_path: str | Path,
    output: str | Path,
    report_output: str | Path = "reports/benchmark_usability_precondition_check.md",
) -> dict[str, Any]:
    config = load_usability_config(config_path)
    out_dir = Path(output)
    _ensure_dir(out_dir)
    current_gate = _gate_json(config["controlling_status"]["current_status_gate"])
    practical_gate = _gate_json(config["controlling_status"]["practical_utility_gate"])
    repair_gate = _gate_json(config["controlling_status"]["repair_signal_role_gate"])
    required_artifacts = [
        config["controlling_status"]["current_status_gate"],
        config["controlling_status"]["practical_utility_gate"],
        config["controlling_status"]["repair_signal_role_gate"],
        config["controlling_status"]["current_manifest"],
        "README.md",
    ]
    missing = [path for path in required_artifacts if not Path(path).exists() or Path(path).stat().st_size == 0]
    source_hashes = {
        path: {"sha256": _sha256(path), "bytes": Path(path).stat().st_size}
        for path in PROTECTED_ARTIFACTS
        if Path(path).exists()
    }
    _write_json(out_dir / "source_artifact_hashes.json", {"artifacts": source_hashes})
    dependency_scan = _scan_forbidden_runtime_refs([Path("src"), Path("scripts"), Path("examples")])
    artifact_refs = yaml.safe_dump(config["controlling_status"]).lower()
    forbidden_evidence_refs = []
    if "heat_exchanger" in artifact_refs:
        forbidden_evidence_refs.append("heat_exchanger referenced as evidence")
    if "rssm" in artifact_refs:
        forbidden_evidence_refs.append("RSSM referenced as evidence")
    surface_dirs = [path for path in FORBIDDEN_SURFACE_DIRS if Path(path).exists()]
    readme = Path("README.md").read_text(encoding="utf-8")
    reasons: list[str] = []
    if missing:
        reasons.append(f"missing required artifacts: {missing}")
    if current_gate.get("decision") != "CURRENT_STATUS_SYNCED":
        reasons.append("current status gate is not CURRENT_STATUS_SYNCED")
    if practical_gate.get("decision") != "NARROW_TO_WEAK_LOW_COVERAGE_CLAIM":
        reasons.append("practical utility gate is not NARROW_TO_WEAK_LOW_COVERAGE_CLAIM")
    if repair_gate.get("decision") != "MARK_REPAIR_DIAGNOSTIC_ONLY_FOR_CSTR":
        reasons.append("repair role gate is not MARK_REPAIR_DIAGNOSTIC_ONLY_FOR_CSTR")
    if config["expansion_policy"]["scientific_expansion_allowed"] is not False:
        reasons.append("scientific expansion is allowed")
    if config["expansion_policy"]["usability_expansion_allowed"] is not True:
        reasons.append("usability expansion is not allowed")
    if README_START not in readme or README_END not in readme:
        reasons.append("README current status marker is missing")
    if dependency_scan["old_repo_runtime_import_hits"] or dependency_scan["path_hack_hits"]:
        reasons.append("forbidden runtime dependency/path scan failed")
    if forbidden_evidence_refs:
        reasons.extend(forbidden_evidence_refs)
    if surface_dirs:
        reasons.append(f"forbidden product/API/frontend directories exist: {surface_dirs}")
    verdict = "READY_FOR_BENCHMARK_USABILITY_EXPANSION" if not reasons else "NOT_READY"
    result = {
        "package_id": config["package_id"],
        "working_tree_dirty": bool(_git_dirty_lines()),
        "dirty_state": _git_dirty_lines(),
        "current_status_gate": current_gate.get("decision"),
        "practical_utility_gate": practical_gate.get("decision"),
        "repair_signal_role_gate": repair_gate.get("decision"),
        "allowed_claim": config["allowed_claim"]["text"],
        "scientific_expansion_allowed": config["expansion_policy"]["scientific_expansion_allowed"],
        "usability_expansion_allowed": config["expansion_policy"]["usability_expansion_allowed"],
        "required_artifacts": [{"path": path, "exists": path not in missing} for path in required_artifacts],
        "forbidden_dependency_scan": dependency_scan,
        "forbidden_evidence_refs": forbidden_evidence_refs,
        "forbidden_surface_dirs": surface_dirs,
        "source_artifact_hash_manifest": str(out_dir / "source_artifact_hashes.json"),
        "verdict": verdict,
        "reasons": reasons,
    }
    _write_json(out_dir / "precondition_check.json", result)
    write_usability_precondition_report(result, report_output)
    if verdict != "READY_FOR_BENCHMARK_USABILITY_EXPANSION":
        raise RuntimeError(f"benchmark usability preconditions failed: {reasons}")
    return result


def write_usability_precondition_report(result: dict[str, Any], output: str | Path) -> None:
    artifacts = pd.DataFrame(result["required_artifacts"])
    scan = result["forbidden_dependency_scan"]
    text = f"""# Benchmark Usability Preconditions

## Working tree

Dirty: {result["working_tree_dirty"]}

## Current evidence status

Current status gate: {result["current_status_gate"]}

Practical utility gate: {result["practical_utility_gate"]}

Repair signal role gate: {result["repair_signal_role_gate"]}

Allowed claim: {result["allowed_claim"]}

## Expansion policy

Scientific expansion allowed: {result["scientific_expansion_allowed"]}

Usability expansion allowed: {result["usability_expansion_allowed"]}

## Required artifacts

{_markdown_table(artifacts, ["path", "exists"])}

## Forbidden dependency scan

Old repo runtime import hits: {scan["old_repo_runtime_import_hits"] or "none"}

Path hack hits: {scan["path_hack_hits"] or "none"}

Forbidden evidence refs: {result["forbidden_evidence_refs"] or "none"}

Forbidden surface dirs: {result["forbidden_surface_dirs"] or "none"}

## Source artifact hashes

{result["source_artifact_hash_manifest"]}

## Verdict

{result["verdict"]}
"""
    target = Path(output)
    _ensure_dir(target.parent)
    target.write_text(text, encoding="utf-8")


def _fit_models(model_ids: list[str], train: TrajectoryBatch, seed: int) -> dict[str, SimulatorModel]:
    models: dict[str, SimulatorModel] = {}
    for idx, model_id in enumerate(model_ids):
        model = make_model(model_id, seed=seed + idx)
        model.fit(train)
        models[model_id] = model
    return models


def _load_custom_model(spec: str) -> SimulatorModel:
    path_text, class_name = spec.split(":", 1)
    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError(path)
    module_name = f"_scs_custom_model_{hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:10]}"
    module_spec = importlib.util.spec_from_file_location(module_name, path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"could not import custom model from {spec}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    module_spec.loader.exec_module(module)
    model_cls = getattr(module, class_name)
    model = model_cls()
    if not hasattr(model, "fit") or not hasattr(model, "predict_rollout"):
        raise TypeError("custom model must implement fit and predict_rollout")
    return model


def _support_scores(train: TrajectoryBatch, batch: TrajectoryBatch) -> np.ndarray:
    support = SupportDistance()
    support.fit(train)
    return support.score_batch(batch)


def _prediction_rows(
    system_id: str,
    dt: float,
    train: TrajectoryBatch,
    eval_batches: dict[str, TrajectoryBatch],
    models: dict[str, SimulatorModel],
    is_custom: dict[str, bool],
    seed: int,
    uncertainty_samples: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    system = make_system(system_id)
    support_by_split = {split: _support_scores(train, batch) for split, batch in eval_batches.items()}
    rows = []
    prediction_cache: dict[tuple[str, str, int], np.ndarray] = {}
    for split, batch in eval_batches.items():
        for model_id, model in models.items():
            for idx in range(batch.n_trajectories):
                initial = batch.states[idx, 0]
                actions = batch.actions[idx]
                disturbances = batch.disturbances[idx]
                pred = np.asarray(model.predict_rollout(initial, actions, disturbances), dtype=float)
                expected_shape = batch.states[idx].shape
                if pred.shape != expected_shape:
                    raise ValueError(f"{model_id} predicted shape {pred.shape}, expected {expected_shape}")
                if not np.isfinite(pred).all():
                    raise ValueError(f"{model_id} produced non-finite predictions")
                prediction_cache[(model_id, split, idx)] = pred
    for split, batch in eval_batches.items():
        for idx in range(batch.n_trajectories):
            all_preds = [prediction_cache[(model_id, split, idx)] for model_id in models]
            scenario_disagreement = disagreement_score(all_preds)
            for model_id, model in models.items():
                pred = prediction_cache[(model_id, split, idx)]
                truth = batch.states[idx]
                try:
                    uncert = uncertainty_score(
                        model,
                        batch.states[idx, 0],
                        batch.actions[idx],
                        batch.disturbances[idx],
                        n_samples=uncertainty_samples,
                    )
                except Exception:
                    uncert = 0.0
                rows.append(
                    {
                        "scenario_id": f"{split}_{idx:04d}",
                        "model_id": model_id,
                        "system_id": system_id,
                        "split": split,
                        "scenario_type": batch.scenario_type[idx],
                        "rmse": rmse(pred, truth),
                        "mae": mae(pred, truth),
                        "max_abs_error": max_abs_error(pred, truth),
                        "final_state_error": final_state_error(pred, truth),
                        "support_distance": float(support_by_split[split][idx]),
                        "uncertainty": float(uncert),
                        "disagreement": float(scenario_disagreement),
                        "invariant_residual": invariant_residual_score(system, pred, batch.actions[idx], batch.disturbances[idx], dt),
                        "repair_amount": repair_amount_score(system, pred),
                        "error": rmse(pred, truth),
                        "is_builtin": not is_custom[model_id],
                        "is_custom": is_custom[model_id],
                    }
                )
    scenario = pd.DataFrame(rows)
    judge_scores = compute_judge_score_frame(
        scenario[["support_distance", "uncertainty", "disagreement", "invariant_residual", "repair_amount", "error"]],
        ["support_only", "combined_linear", "oracle_error_rank"],
        seed=seed,
    )
    for column in judge_scores:
        scenario[f"risk_{column}"] = judge_scores[column].to_numpy(dtype=float)
    metrics = (
        scenario.groupby(["model_id", "system_id", "split", "is_builtin", "is_custom"], as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            mae_mean=("mae", "mean"),
            max_abs_error_mean=("max_abs_error", "mean"),
            final_state_error_mean=("final_state_error", "mean"),
        )
        .sort_values(["model_id", "split"])
    )
    return scenario, metrics


def _risk_by_model(scenario: pd.DataFrame, coverages: list[float], threshold: float) -> pd.DataFrame:
    rows = []
    for (model_id, split), frame in scenario.groupby(["model_id", "split"]):
        curve = risk_coverage_curve(
            frame["error"].to_numpy(dtype=float),
            frame["risk_support_only"].to_numpy(dtype=float),
            bad_threshold=threshold,
            coverages=coverages,
        )
        curve.insert(0, "model_id", model_id)
        curve.insert(1, "split", split)
        curve.insert(2, "judge_id", "support_only_local")
        rows.append(curve)
    return pd.concat(rows, ignore_index=True)


def _plot_model_risk(risk: pd.DataFrame, output: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    low = risk[np.isclose(risk["coverage"].astype(float), float(risk["coverage"].min()))]
    for model_id, frame in low.groupby("model_id"):
        ax.bar(model_id, frame["false_accept_rate"].mean())
    ax.set_title("Local low-coverage FAR by model")
    ax.set_ylabel("False accept rate")
    ax.set_xlabel("Model")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def run_current_status_demo(
    config_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    config = load_usability_config(config_path)
    manifest = _read_json(config["controlling_status"]["current_manifest"])
    out_dir = Path(output)
    _ensure_dir(out_dir)
    dataset = generate_dataset(
        system_id="two_tank",
        n_train=48,
        n_id_test=12,
        n_ood_test=12,
        horizon=20,
        dt=0.1,
        seed=101,
    )
    model_ids = ["hold_last", "linear_narx", "mlp_state_space"]
    models = _fit_models(model_ids, dataset["train"], seed=101)
    eval_batches = {key: dataset[key] for key in ["id_test", "ood_action_magnitude", "ood_inflow_spike", "ood_combined"]}
    scenario, metrics = _prediction_rows(
        "two_tank",
        0.1,
        dataset["train"],
        eval_batches,
        models,
        {model_id: False for model_id in models},
        seed=101,
        uncertainty_samples=2,
    )
    best_model = (
        metrics[metrics["split"] == "id_test"]
        .sort_values("rmse_mean")
        .iloc[0]["model_id"]
    )
    demo_frame = scenario[scenario["model_id"] == best_model].copy()
    coverages = [0.05, 0.10]
    rows = []
    risk_plot_rows = []
    for coverage in coverages:
        baseline = risk_coverage_curve(
            demo_frame["error"].to_numpy(dtype=float),
            demo_frame["risk_support_only"].to_numpy(dtype=float),
            bad_threshold=0.15,
            coverages=[coverage],
        ).iloc[0]
        calibrated = risk_coverage_curve(
            demo_frame["error"].to_numpy(dtype=float),
            demo_frame["risk_combined_linear"].to_numpy(dtype=float),
            bad_threshold=0.15,
            coverages=[coverage],
        ).iloc[0]
        rows.append(
            {
                "system_id": "two_tank",
                "coverage": float(coverage),
                "baseline_judge": "support_only_demo",
                "calibrated_judge": "combined_linear_demo",
                "baseline_far": float(baseline["false_accept_rate"]),
                "calibrated_far": float(calibrated["false_accept_rate"]),
                "absolute_margin": float(baseline["false_accept_rate"] - calibrated["false_accept_rate"]),
                "claim_scope": "demo_only_not_full_evidence",
                "is_demo": True,
            }
        )
        risk_plot_rows.extend(
            [
                {"coverage": coverage, "judge": "support_only_demo", "false_accept_rate": float(baseline["false_accept_rate"])},
                {"coverage": coverage, "judge": "combined_linear_demo", "false_accept_rate": float(calibrated["false_accept_rate"])},
            ]
        )
    result_table = pd.DataFrame(rows)
    result_table.to_csv(out_dir / "main_result_table.csv", index=False)
    _plot_demo_risk(pd.DataFrame(risk_plot_rows), out_dir / "risk_coverage.png")
    metrics.to_csv(out_dir / "demo_model_metrics.csv", index=False)
    summary = {
        "verdict": "DEMO_BUILT",
        "is_demo": True,
        "best_demo_model_by_id_rmse": str(best_model),
        "allowed_claim": config["allowed_claim"]["text"],
        "current_manifest_status": manifest.get("status_id"),
        "outputs": {
            "main_result_table": str(out_dir / "main_result_table.csv"),
            "risk_coverage": str(out_dir / "risk_coverage.png"),
            "demo_report": str(out_dir / "demo_report.md"),
            "demo_summary": str(out_dir / "demo_summary.json"),
        },
    }
    _write_json(out_dir / "demo_summary.json", summary)
    write_demo_report(result_table, summary, out_dir / "demo_report.md")
    return summary


def _plot_demo_risk(frame: pd.DataFrame, output: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for judge, group in frame.groupby("judge"):
        ax.plot(group["coverage"], group["false_accept_rate"], marker="o", label=judge)
    ax.set_title("Current status demo risk-coverage")
    ax.set_xlabel("Coverage")
    ax.set_ylabel("False accept rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def write_demo_report(result_table: pd.DataFrame, summary: dict[str, Any], output: str | Path) -> None:
    text = f"""# Current Status Demo Report

## What this demo does

Runs a lightweight TwoTank local benchmark path with the built-in models and writes a small risk-coverage result table.

## What this demo does not prove

This demo is not the full evidence chain.
The current supported claim remains weak and low-coverage only.

## Current allowed claim

{summary["allowed_claim"]}

## Demo result table

{_markdown_table(result_table, ["system_id", "coverage", "baseline_judge", "calibrated_judge", "baseline_far", "calibrated_far", "absolute_margin", "claim_scope", "is_demo"])}

## How to reproduce

```bash
python scripts/run_current_status_demo.py --config configs/status/benchmark_usability_v1_1.yaml --output results/demo
```

## Where to find full evidence

See `results/current_status/evidence_manifest/current_evidence_manifest.json` and `reports/current_status_decision_gate.md`.

## Non-claims

This is not safety certification, product readiness, high-coverage reliability, autonomous control, RSSM evidence, or third-system evidence.
"""
    target = Path(output)
    _ensure_dir(target.parent)
    target.write_text(text, encoding="utf-8")


def build_benchmark_card(
    config_path: str | Path,
    manifest_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    config = load_usability_config(config_path)
    manifest = _read_json(manifest_path)
    text = render_benchmark_card(config, manifest)
    target = Path(output)
    _ensure_dir(target.parent)
    target.write_text(text, encoding="utf-8")
    return {"verdict": "BENCHMARK_CARD_BUILT", "output": str(target)}


def render_benchmark_card(config: dict[str, Any], manifest: dict[str, Any]) -> str:
    return f"""# Benchmark Card: Selective Counterfactual Simulation

## Intended use

Use this benchmark to test refusal/ranking behavior for counterfactual simulator rollouts under intervention shift.

## Non-intended use

This benchmark tests refusal/ranking behavior, not simulator safety. It is not intended for product use, certification, autonomous control, or plant-wide deployment claims.

## Core question

Can a simulator identify which counterfactual intervention scenarios it can answer reliably and abstain on the rest?

## Systems included

The current evidence package includes TwoTank and CSTR. TwoTank is stronger than CSTR. Expansion to RSSM, third systems, or product use is not currently supported.

## Models included

Built-in local models are `hold_last`, `linear_narx`, and `mlp_state_space`.

## Refusal signals included

Support distance, uncertainty, disagreement, invariant residual, and repair amount are available. repair_amount is diagnostic-only for CSTR. invariant_residual is informative for CSTR.

## Primary metric

False accept rate at fixed coverage.

## What counts as a false accept

A false accept occurs when a judge accepts a scenario whose simulator prediction is materially wrong under the configured error threshold.

## Current evidence status

{config["allowed_claim"]["text"]} Current evidence is weak-positive and low-coverage only. TwoTank margin at coverage 0.05 is {manifest["systems"]["two_tank"]["coverage_0_05_margin"]:.6f}; CSTR margin at coverage 0.05 is {manifest["systems"]["cstr"]["coverage_0_05_margin"]:.6f}.

## Known weaknesses

CSTR is positive but weak. repair_amount misses within-bound CSTR dynamic errors, while invariant_residual is more informative.

## How to run the quickstart demo

```bash
python scripts/run_current_status_demo.py --config configs/status/benchmark_usability_v1_1.yaml --output results/demo
```

## How to add a custom model

Implement `fit(train_batch)` and `predict_rollout(initial_state, actions, disturbances)` using `src/scs/models/user_model.py`, then run `python examples/custom_model_example.py --output results/custom_model_example`.

## How to compare models fairly

Use the local comparison script and report results as local-only:

```bash
python scripts/compare_models.py --config configs/experiments/calibrated_two_tank.yaml --models hold_last linear_narx mlp_state_space --output results/model_comparison
```

## Claim boundaries

Do not claim strong support, broad simulator reliability, safety certification, product readiness, high-coverage reliability, RSSM evidence, or third-system evidence.

## Reproducibility

Run `pip install -e ".[dev]"`, `pytest -q`, and the quickstart demo command above.
"""


def compare_models(
    config_path: str | Path,
    model_ids: list[str],
    output: str | Path,
    custom_model: str | None = None,
) -> dict[str, Any]:
    config = _read_yaml(config_path)
    for key in ["system_id", "seed", "horizon", "dt"]:
        if key not in config:
            raise ValueError(f"missing comparison config key: {key}")
    system_id = str(config["system_id"])
    if system_id != "two_tank":
        raise ValueError("local usability comparison currently supports TwoTank only")
    out_dir = Path(output)
    _ensure_dir(out_dir)
    seed = int(config.get("seed", 42))
    horizon = int(config.get("horizon", 30))
    dt = float(config.get("dt", 0.1))
    dataset = generate_dataset(
        system_id="two_tank",
        n_train=int(config.get("n_model_train", config.get("n_train", 80))),
        n_id_test=int(config.get("n_test_id", config.get("n_id_test", 30))),
        n_ood_test=int(config.get("n_test_ood", config.get("n_ood_test", 30))),
        horizon=horizon,
        dt=dt,
        seed=seed,
    )
    allowed_builtin = {"hold_last", "linear_narx", "mlp_state_space"}
    unknown = sorted(set(model_ids) - allowed_builtin)
    if unknown:
        raise ValueError(f"unknown built-in model ids: {unknown}")
    models = _fit_models(model_ids, dataset["train"], seed=seed)
    is_custom = {model_id: False for model_id in models}
    if custom_model:
        custom = _load_custom_model(custom_model)
        custom.fit(dataset["train"])
        models[custom.model_id] = custom
        is_custom[custom.model_id] = True
    eval_batches = {key: dataset[key] for key in ["id_test", "ood_action_magnitude", "ood_inflow_spike", "ood_combined"]}
    scenario, metrics = _prediction_rows(
        "two_tank",
        dt,
        dataset["train"],
        eval_batches,
        models,
        is_custom,
        seed=seed,
        uncertainty_samples=int(config.get("uncertainty_samples", 2)),
    )
    threshold = float(config.get("bad_threshold", {}).get("value", 0.15))
    coverages = [float(value) for value in config.get("primary_coverages", [0.05, 0.10])]
    risk = _risk_by_model(scenario, coverages, threshold)
    metrics.to_csv(out_dir / "model_comparison.csv", index=False)
    risk.to_csv(out_dir / "risk_coverage_by_model.csv", index=False)
    _plot_model_risk(risk, out_dir / "risk_coverage_by_model.png")
    summary = {
        "verdict": "MODEL_COMPARISON_BUILT",
        "system_id": "two_tank",
        "models": list(models),
        "is_local_comparison_only": True,
        "custom_model": custom_model,
        "current_claim_modified": False,
        "outputs": {
            "model_comparison": str(out_dir / "model_comparison.csv"),
            "risk_coverage_by_model": str(out_dir / "risk_coverage_by_model.csv"),
            "plot": str(out_dir / "risk_coverage_by_model.png"),
            "summary": str(out_dir / "model_comparison_summary.json"),
            "report": str(out_dir / "model_comparison_report.md"),
        },
    }
    _write_json(out_dir / "model_comparison_summary.json", summary)
    write_model_comparison_report(metrics, risk, summary, out_dir / "model_comparison_report.md")
    return summary


def write_model_comparison_report(metrics: pd.DataFrame, risk: pd.DataFrame, summary: dict[str, Any], output: str | Path) -> None:
    low = risk[np.isclose(risk["coverage"], risk["coverage"].min())]
    best_trajectory = metrics.sort_values("rmse_mean").iloc[0]["model_id"]
    best_far = low.groupby("model_id")["false_accept_rate"].mean().sort_values().index[0]
    text = f"""# Local Model Comparison Report

## Command

Local comparison generated by `scripts/compare_models.py`.

## Models compared

{", ".join(summary["models"])}

## System

{summary["system_id"]}

## Trajectory metrics

{_markdown_table(metrics, ["model_id", "system_id", "split", "rmse_mean", "mae_mean", "max_abs_error_mean", "final_state_error_mean", "is_builtin", "is_custom"])}

## Low-coverage refusal metrics

{_markdown_table(low, ["model_id", "split", "coverage", "judge_id", "false_accept_rate", "accepted_count", "false_accept_count"])}

## Best model by trajectory error

{best_trajectory}

## Best model by low-coverage FAR

{best_far}

## Important caveat

These results are local comparison results and do not modify the current supported claim.
"""
    target = Path(output)
    _ensure_dir(target.parent)
    target.write_text(text, encoding="utf-8")


def render_readme_usability_block(config: dict[str, Any]) -> str:
    return f"""{USABILITY_START}
## Quickstart

```bash
pip install -e ".[dev]"
pytest -q
python scripts/run_current_status_demo.py
```

## Run the Current Status Demo

```bash
python scripts/run_current_status_demo.py --config configs/status/benchmark_usability_v1_1.yaml --output results/demo
```

The demo is a quick local run, not the full evidence chain.

## What This Benchmark Tests

It tests refusal/ranking behavior for counterfactual simulator rollouts under intervention shift.

## What This Benchmark Does Not Test

It does not test simulator safety, product readiness, plant-wide deployment, RSSM evidence, third-system evidence, autonomous control, or high-coverage reliability.

## Add Your Own Model

Implement the adapter in `src/scs/models/user_model.py`, inspect `examples/custom_model_example.py`, and run:

```bash
python examples/custom_model_example.py --output results/custom_model_example
```

## Local Model Comparison

```bash
python scripts/compare_models.py --config configs/experiments/calibrated_two_tank.yaml --models hold_last linear_narx mlp_state_space --output results/model_comparison
```

Custom model example:

```bash
python scripts/compare_models.py --config configs/experiments/calibrated_two_tank.yaml --models linear_narx mlp_state_space --custom-model examples/custom_model_example.py:DampedLinearUserModel --output results/model_comparison_custom
```

## Current Evidence Status

{config["allowed_claim"]["text"]} TwoTank is stronger than CSTR. repair_amount is diagnostic-only for CSTR; invariant_residual is informative for CSTR.

## Reproducibility

Run the install, test, demo, and comparison commands above from the repository root.

## Claim Boundaries

This usability release does not change the scientific claim. It does not add RSSM, third-system evidence, new benchmark systems, product API, frontend, or deployment work.
{USABILITY_END}"""


def update_readme_usability_sections(config_path: str | Path, readme_path: str | Path, check: bool = False) -> dict[str, Any]:
    config = load_usability_config(config_path)
    readme = Path(readme_path)
    write_sync_artifact = readme.resolve() == Path("README.md").resolve()
    current = readme.read_text(encoding="utf-8")
    block = render_readme_usability_block(config)
    if USABILITY_START in current and USABILITY_END in current:
        import re

        updated = re.sub(re.escape(USABILITY_START) + r".*?" + re.escape(USABILITY_END), block, current, flags=re.DOTALL)
    else:
        insert_at = current.find(README_END)
        if insert_at >= 0:
            insert_at += len(README_END)
            updated = current[:insert_at] + "\n\n" + block + current[insert_at:]
        else:
            updated = current + "\n\n" + block + "\n"
    stale = updated != current
    if check and stale:
        result = {"verdict": "README_USABILITY_STALE", "stale": True, "readme": str(readme)}
        if write_sync_artifact:
            _write_json(USABILITY_RESULTS_ROOT / "readme_usability_sync.json", result)
        raise RuntimeError("README usability sections are stale")
    if not check:
        readme.write_text(updated, encoding="utf-8")
        stale = False
    text = readme.read_text(encoding="utf-8")
    missing = [section for section in REQUIRED_README_SECTIONS if section not in text]
    result = {
        "verdict": "README_USABILITY_SYNCED" if not stale and not missing else "README_USABILITY_STALE",
        "stale": stale,
        "missing_sections": missing,
        "readme": str(readme),
    }
    if write_sync_artifact:
        _write_json(USABILITY_RESULTS_ROOT / "readme_usability_sync.json", result)
    if result["verdict"] != "README_USABILITY_SYNCED":
        raise RuntimeError(f"README usability sync failed: {result}")
    return result


def build_benchmark_usability_release(config_path: str | Path, output: str | Path) -> dict[str, Any]:
    config = load_usability_config(config_path)
    out_dir = Path(output)
    _ensure_dir(out_dir)
    manifest = {
        "release_name": "v1.1-benchmark-usability",
        "release_type": "usability_only",
        "scientific_claim_changed": False,
        "allowed_claim": config["allowed_claim"]["text"],
        "new_user_features": config["allowed_usability_additions"],
        "forbidden_claims": config["forbidden_claims"],
        "demo_commands": [
            "python scripts/run_current_status_demo.py --config configs/status/benchmark_usability_v1_1.yaml --output results/demo",
        ],
        "custom_model_interface": "src/scs/models/user_model.py",
        "model_comparison_script": "scripts/compare_models.py",
        "known_limitations": [
            "Scientific claim unchanged.",
            "Current evidence remains weak-positive and low-coverage only.",
            "Expansion remains blocked.",
            "Custom model outputs are local-only and not benchmark evidence.",
        ],
    }
    _write_json(out_dir / "benchmark_usability_manifest.json", manifest)
    write_benchmark_usability_release_note(manifest, "reports/release_note_v1_1_benchmark_usability.md")
    return manifest


def write_benchmark_usability_release_note(manifest: dict[str, Any], output: str | Path) -> None:
    text = f"""# Release Note: v1.1 Benchmark Usability

## What changed

Added a quickstart demo, benchmark card, custom model adapter, custom model example, local model comparison, README usability sections, and a usability package checker.

## What did not change

The scientific claim did not change. Expansion remains blocked.

## Current allowed claim

{manifest["allowed_claim"]}

## New quickstart demo

Run `python scripts/run_current_status_demo.py --config configs/status/benchmark_usability_v1_1.yaml --output results/demo`.

## Custom model adapter

Use `src/scs/models/user_model.py` and `examples/custom_model_example.py`.

## Local model comparison

Use `scripts/compare_models.py` for built-in or custom models. Results are local-only.

## Claim boundaries

Do not claim strong support, safety certification, product readiness, high-coverage reliability, RSSM evidence, or third-system evidence.

## Reproducibility

Run `pip install -e ".[dev]"`, `pytest -q`, the demo command, and the model comparison command from the README.

## What not to do next

Do not treat usability features as new benchmark evidence. Do not add RSSM, third-system, product/API/frontend, or deployment claims.
"""
    target = Path(output)
    _ensure_dir(target.parent)
    target.write_text(text, encoding="utf-8")


def check_benchmark_usability_package(config_path: str | Path) -> dict[str, Any]:
    config = load_usability_config(config_path)
    temp_root = USABILITY_RESULTS_ROOT / "package_check_runs"
    demo_temp = temp_root / "demo"
    compare_temp = temp_root / "model_comparison"
    custom_compare_temp = temp_root / "model_comparison_custom"
    custom_example_temp = temp_root / "custom_model_example"
    demo_summary = run_current_status_demo(config_path, demo_temp)
    compare_summary = compare_models(
        "configs/experiments/calibrated_two_tank.yaml",
        ["hold_last", "linear_narx"],
        compare_temp,
    )
    custom_summary = compare_models(
        "configs/experiments/calibrated_two_tank.yaml",
        ["linear_narx"],
        custom_compare_temp,
        custom_model=config["custom_model"]["example_path"] + ":DampedLinearUserModel",
    )
    custom_run = subprocess.run(
        [sys.executable, config["custom_model"]["example_path"], "--output", str(custom_example_temp)],
        check=False,
        capture_output=True,
        text=True,
    )
    readme_check = update_readme_usability_sections(config_path, "README.md", check=True)
    claim_scan = scan_forbidden_claim_language(
        [
            "README.md",
            "docs/benchmark_card.md",
            "docs/custom_model_adapter.md",
            "reports/release_note_v1_1_benchmark_usability.md",
        ],
        config["forbidden_claims"],
    )
    precondition_hashes = _read_json(USABILITY_RESULTS_ROOT / "preconditions" / "source_artifact_hashes.json")["artifacts"]
    hash_mismatches = [
        path
        for path, item in precondition_hashes.items()
        if Path(path).exists() and _sha256(path) != item["sha256"]
    ]
    required_files = [
        "results/benchmark_usability/preconditions/precondition_check.json",
        "results/demo/main_result_table.csv",
        "results/demo/risk_coverage.png",
        "results/demo/demo_report.md",
        "results/demo/demo_summary.json",
        "docs/benchmark_card.md",
        "src/scs/models/user_model.py",
        "examples/custom_model_example.py",
        "docs/custom_model_adapter.md",
        "results/model_comparison/model_comparison.csv",
        "results/model_comparison/risk_coverage_by_model.csv",
        "results/model_comparison/risk_coverage_by_model.png",
        "results/model_comparison/model_comparison_summary.json",
        "results/model_comparison/model_comparison_report.md",
        "results/model_comparison_custom/model_comparison.csv",
        "results/model_comparison_custom/risk_coverage_by_model.csv",
        "results/model_comparison_custom/risk_coverage_by_model.png",
        "results/model_comparison_custom/model_comparison_summary.json",
        "results/model_comparison_custom/model_comparison_report.md",
        "results/benchmark_usability/release/benchmark_usability_manifest.json",
        "reports/release_note_v1_1_benchmark_usability.md",
    ]
    missing = [path for path in required_files if not Path(path).exists() or Path(path).stat().st_size == 0]
    reasons = []
    if missing:
        reasons.append(f"missing files: {missing}")
    if demo_summary.get("verdict") != "DEMO_BUILT":
        reasons.append("demo check failed")
    if compare_summary.get("verdict") != "MODEL_COMPARISON_BUILT" or custom_summary.get("verdict") != "MODEL_COMPARISON_BUILT":
        reasons.append("model comparison check failed")
    if custom_run.returncode != 0:
        reasons.append(f"custom model example failed: {custom_run.stderr}")
    if readme_check.get("verdict") != "README_USABILITY_SYNCED":
        reasons.append("README check failed")
    if claim_scan["violations"]:
        reasons.append("claim language violations")
    if hash_mismatches:
        reasons.append(f"prior artifact hash mismatches: {hash_mismatches}")
    verdict = "BENCHMARK_USABILITY_PACKAGE_ACCEPTED" if not reasons else "BENCHMARK_USABILITY_PACKAGE_REJECTED"
    result = {
        "verdict": verdict,
        "demo_check": demo_summary.get("verdict"),
        "model_comparison_check": compare_summary.get("verdict"),
        "custom_model_comparison_check": custom_summary.get("verdict"),
        "custom_model_example_returncode": custom_run.returncode,
        "readme_check": readme_check.get("verdict"),
        "claim_language_violations": claim_scan["violations"],
        "prior_artifact_mutation_detected": bool(hash_mismatches),
        "source_hash_mismatches": hash_mismatches,
        "missing": missing,
        "reasons": reasons,
    }
    _write_json(USABILITY_RESULTS_ROOT / "package_check.json", result)
    write_benchmark_usability_package_check_report(result, "reports/benchmark_usability_package_check.md")
    if verdict != "BENCHMARK_USABILITY_PACKAGE_ACCEPTED":
        raise RuntimeError(f"benchmark usability package rejected: {reasons}")
    return result


def write_benchmark_usability_package_check_report(result: dict[str, Any], output: str | Path) -> None:
    text = f"""# Benchmark Usability Package Check

## Demo

{result["demo_check"]}

## Model comparison

Built-in: {result["model_comparison_check"]}

Custom: {result["custom_model_comparison_check"]}

## Custom model example

Return code: {result["custom_model_example_returncode"]}

## README check

{result["readme_check"]}

## Claim language violations

{result["claim_language_violations"] or "none"}

## Prior artifact mutation detected

{result["prior_artifact_mutation_detected"]}

## Missing files

{result["missing"] or "none"}

## Verdict

{result["verdict"]}
"""
    target = Path(output)
    _ensure_dir(target.parent)
    target.write_text(text, encoding="utf-8")
