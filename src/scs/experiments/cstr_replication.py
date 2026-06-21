from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scs.experiments.calibrated import (
    CSTR_REQUIRED_SCENARIO_TYPES,
    _ensure_dir,
    _load_json,
    _markdown_table,
    _score_scenarios,
    _write_json,
    generate_calibrated_data,
    load_calibrated_config,
)
from scs.experiments.registry import make_model
from scs.systems.cstr import CSTRSystem
from scs.validators.support import SupportDistance


PROTOCOL_SECTIONS = [
    "Source checkpoint",
    "Purpose",
    "Candidate judges",
    "Real baseline judges",
    "Diagnostic-only judges",
    "Signal columns",
    "Models",
    "Data roles",
    "Calibration rules",
    "Primary coverages",
    "Full coverage grid",
    "Bad-threshold grid",
    "Seed-sweep rules",
    "Stress-test rules",
    "CSTR decision rules",
    "Multi-system decision rules",
    "Forbidden changes",
    "Allowed CSTR-specific changes",
    "Leakage rules",
    "Report verdicts",
]
FROZEN_CANDIDATE_JUDGES = [
    "best_single_signal_selected_on_calibration",
    "rank_normalized_linear",
    "logistic_calibrated_judge",
    "isotonic_calibrated_judge",
    "quantile_rule_judge",
    "conservative_low_coverage_judge",
    "calibration_selected_candidate_ranker",
]
FROZEN_BASELINE_JUDGES = [
    "support_only",
    "uncertainty_only",
    "disagreement_only",
    "invariant_only",
    "repair_only",
    "combined_linear",
    "random_baseline",
]
FROZEN_SIGNALS = [
    "support_distance",
    "uncertainty_score",
    "disagreement_score",
    "invariant_residual",
    "repair_amount",
]
FROZEN_MODELS = ["hold_last", "linear_narx", "mlp_state_space"]
FROZEN_PRIMARY_COVERAGES = [0.05, 0.10]
FROZEN_COVERAGES = [0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00]
FROZEN_THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
OLD_REPO_NAMES = [
    "time" + "-series" + "-simulator",
    "digital" + "-twin" + "-engine",
    "flux" + "-attention" + "-engine",
    "plant" + "-scenario" + "-compiler",
]


def _current_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _section_body(text: str, section: str) -> str:
    pattern = rf"^## {re.escape(section)}\n(?P<body>.*?)(?=^## |\Z)"
    match = re.search(pattern, text, flags=re.MULTILINE | re.DOTALL)
    return "" if match is None else match.group("body")


def _listed_items(text: str, section: str) -> list[str]:
    body = _section_body(text, section)
    return [line[2:].strip() for line in body.splitlines() if line.startswith("- ")]


def validate_protocol_lock(protocol: str | Path) -> dict[str, Any]:
    path = Path(protocol)
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    missing_sections = [section for section in PROTOCOL_SECTIONS if f"## {section}" not in text]
    candidate_judges = _listed_items(text, "Candidate judges")
    baseline_judges = _listed_items(text, "Real baseline judges")
    signals = _listed_items(text, "Signal columns")
    models = _listed_items(text, "Models")
    primary = [float(value) for value in re.findall(r"^- ([0-9]+\.[0-9]+)$", _section_body(text, "Primary coverages"), re.MULTILINE)]
    coverage = [float(value) for value in re.findall(r"^- ([0-9]+\.[0-9]+)$", _section_body(text, "Full coverage grid"), re.MULTILINE)]
    thresholds = [float(value) for value in re.findall(r"^- ([0-9]+\.[0-9]+)$", _section_body(text, "Bad-threshold grid"), re.MULTILINE)]
    result = {
        "path": str(path),
        "missing_sections": missing_sections,
        "candidate_judges": candidate_judges,
        "baseline_judges": baseline_judges,
        "signals": signals,
        "models": models,
        "primary_coverages": primary,
        "coverages": coverage,
        "thresholds": thresholds,
        "candidate_judges_match": candidate_judges == FROZEN_CANDIDATE_JUDGES,
        "baseline_judges_match": baseline_judges == FROZEN_BASELINE_JUDGES,
        "signals_match": signals == FROZEN_SIGNALS,
        "models_match": models == FROZEN_MODELS,
        "primary_coverages_match": primary == FROZEN_PRIMARY_COVERAGES,
        "coverages_match": coverage == FROZEN_COVERAGES,
        "thresholds_match": thresholds == FROZEN_THRESHOLDS,
        "states_replication_only": "CSTR is replication evidence only" in text,
    }
    result["valid"] = bool(
        not missing_sections
        and result["candidate_judges_match"]
        and result["baseline_judges_match"]
        and result["signals_match"]
        and result["models_match"]
        and result["primary_coverages_match"]
        and result["coverages_match"]
        and result["thresholds_match"]
        and result["states_replication_only"]
    )
    return result


def _scan_forbidden_runtime_imports() -> dict[str, list[str]]:
    old_repo_hits = []
    path_hacks = []
    for root in [Path("src"), Path("scripts"), Path("tests")]:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            lines = path.read_text(encoding="utf-8").splitlines()
            for line in lines:
                stripped = line.strip()
                if stripped.startswith(("import ", "from ")) and any(name in stripped for name in OLD_REPO_NAMES):
                    old_repo_hits.append(str(path))
                    break
            for line in lines:
                stripped = line.strip()
                env_key = "PYTHON" + "PATH"
                if stripped.startswith("sys.path") or stripped.startswith(f"os.environ[\"{env_key}\""):
                    path_hacks.append(str(path))
                    break
    return {"old_repo_runtime_import_hits": sorted(set(old_repo_hits)), "path_hack_hits": sorted(set(path_hacks))}


def verify_cstr_preconditions(protocol: str | Path, output: str | Path) -> dict[str, Any]:
    out_dir = Path(output)
    _ensure_dir(out_dir)
    protocol_check = validate_protocol_lock(protocol)
    gate_path = Path("reports/calibrated_judge_decision_gate.json")
    gate = _load_json(gate_path) if gate_path.exists() else {}
    cstr_impl = Path("src/scs/systems/cstr.py").exists()
    cstr_config = Path("configs/experiments/calibrated_cstr.yaml").exists()
    forbidden_scan = _scan_forbidden_runtime_imports()
    reasons = []
    if not protocol_check["valid"]:
        reasons.append("protocol lock is invalid")
    if gate.get("decision") != "PROCEED_TO_CSTR":
        reasons.append("previous calibrated gate did not allow CSTR")
    if not cstr_impl:
        reasons.append("CSTR implementation missing")
    if not cstr_config:
        reasons.append("CSTR calibrated config missing")
    if forbidden_scan["old_repo_runtime_import_hits"] or forbidden_scan["path_hack_hits"]:
        reasons.append("forbidden dependency scan failed")
    verdict = "READY_FOR_CSTR_SANITY" if not reasons else "NOT_READY"
    result = {
        "protocol_lock": protocol_check,
        "previous_calibrated_gate": gate,
        "cstr_implementation_exists": cstr_impl,
        "cstr_config_exists": cstr_config,
        "forbidden_dependency_scan": forbidden_scan,
        "allowed_expansion": ["CSTR"],
        "forbidden_evidence": ["RSSM", "heat_exchanger", "old repos"],
        "git_commit_at_check": _current_git_commit(),
        "verdict": verdict,
        "reasons": reasons,
    }
    _write_json(out_dir / "precondition_check.json", result)
    report = f"""# CSTR Preconditions

## Protocol lock

Valid: {protocol_check["valid"]}

## Previous calibrated gate

{gate.get("decision", "missing")}

## CSTR implementation status

Implementation exists: {cstr_impl}
Config exists: {cstr_config}

## Forbidden dependency scan

Old repo hits: {forbidden_scan["old_repo_runtime_import_hits"] or "none"}
Path hack hits: {forbidden_scan["path_hack_hits"] or "none"}

## Expansion status

Allowed expansion: CSTR only. RSSM and heat_exchanger evidence are forbidden.

## Verdict

{verdict}
"""
    report_path = Path("reports/cstr_precondition_check.md")
    _ensure_dir(report_path.parent)
    report_path.write_text(report, encoding="utf-8")
    if verdict != "READY_FOR_CSTR_SANITY":
        raise RuntimeError(f"CSTR preconditions not ready: {reasons}")
    return result


def _range(values: np.ndarray) -> float:
    return float(np.max(values) - np.min(values))


def _distribution_checks(dataset: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    id_batch = dataset["judge_calibration_id"]
    id_action_range = _range(id_batch.actions)
    id_d0_max = float(np.max(id_batch.disturbances[..., 0]))
    id_d1_max = float(np.max(id_batch.disturbances[..., 1]))
    id_d2_max = float(np.max(id_batch.disturbances[..., 2]))
    for split, batch in dataset.items():
        rows.append(
            {
                "split": split,
                "role": "model_train" if split == "model_train" else ("judge_calibration" if split.startswith("judge_calibration") else "judge_test"),
                "scenario_type": ",".join(sorted(set(batch.scenario_type))),
                "action_min": float(np.min(batch.actions)),
                "action_max": float(np.max(batch.actions)),
                "action_range": _range(batch.actions),
                "disturbance_0_max": float(np.max(batch.disturbances[..., 0])),
                "disturbance_1_max": float(np.max(batch.disturbances[..., 1])),
                "disturbance_2_max": float(np.max(batch.disturbances[..., 2])),
                "action_range_ratio_vs_id": _range(batch.actions) / max(id_action_range, 1e-12),
                "disturbance_0_ratio_vs_id": float(np.max(batch.disturbances[..., 0])) / max(id_d0_max, 1e-12),
                "disturbance_1_delta_vs_id": float(np.max(batch.disturbances[..., 1]) - id_d1_max),
                "disturbance_2_ratio_vs_id": float(np.max(batch.disturbances[..., 2])) / max(id_d2_max, 1e-12),
            }
        )
    table = pd.DataFrame(rows)
    checks = {
        "cooling_action_shift": float(
            table.loc[table["split"] == "judge_calibration_cooling_step_change", "action_range_ratio_vs_id"].iloc[0]
        ),
        "feed_concentration_shift": float(
            table.loc[
                table["split"] == "judge_calibration_feed_concentration_spike",
                "disturbance_0_ratio_vs_id",
            ].iloc[0]
        ),
        "feed_temperature_shift": float(
            table.loc[
                table["split"] == "judge_calibration_feed_temperature_spike",
                "disturbance_1_delta_vs_id",
            ].iloc[0]
        ),
    }
    return table, checks


def _finite_and_nonconstant(dataset: dict[str, Any]) -> tuple[bool, bool]:
    finite = True
    nonconstant = True
    for batch in dataset.values():
        finite = finite and bool(np.isfinite(batch.states).all() and np.isfinite(batch.actions).all() and np.isfinite(batch.disturbances).all())
        changes = np.linalg.norm(np.diff(batch.states, axis=1), axis=-1)
        nonconstant = nonconstant and bool(np.mean(changes > 1e-8) > 0.5)
    return finite, nonconstant


def _physically_plausible(dataset: dict[str, Any]) -> bool:
    system = CSTRSystem()
    for batch in dataset.values():
        concentration = batch.states[..., 0]
        temperature = batch.states[..., 1]
        if float(np.min(concentration)) < system.concentration_bounds[0] - 1e-9:
            return False
        if float(np.max(concentration)) > system.concentration_bounds[1] + 1e-9:
            return False
        if float(np.min(temperature)) < system.temperature_bounds[0] - 1e-9:
            return False
        if float(np.max(temperature)) > system.temperature_bounds[1] + 1e-9:
            return False
    return True


def _label_checks(config: dict[str, Any], calibration_table: pd.DataFrame, test_table: pd.DataFrame) -> dict[str, Any]:
    threshold = float(config["bad_threshold"]["value"])
    rows = []
    for name, table in [("judge_calibration", calibration_table), ("judge_test", test_table)]:
        labels = table["rmse"].to_numpy(dtype=float) > threshold
        rows.append(
            {
                "role": name,
                "row_count": int(len(table)),
                "bad_count": int(labels.sum()),
                "bad_rate": float(labels.mean()),
                "non_degenerate": bool(labels.any() and (~labels).any()),
            }
        )
    scenario_rows = []
    for name, table in [("judge_calibration", calibration_table), ("judge_test", test_table)]:
        for scenario_type, group in table.groupby("scenario_type", sort=True):
            labels = group["rmse"].to_numpy(dtype=float) > threshold
            scenario_rows.append(
                {
                    "role": name,
                    "scenario_type": str(scenario_type),
                    "row_count": int(len(group)),
                    "bad_rate": float(labels.mean()),
                }
            )
    return {
        "threshold": threshold,
        "roles": rows,
        "scenario_bad_rates": scenario_rows,
        "labels_non_degenerate": all(row["non_degenerate"] for row in rows),
    }


def _model_error_checks(test_table: pd.DataFrame) -> tuple[pd.DataFrame, bool, bool]:
    rows = []
    for model_id, group in test_table.groupby("model_id", sort=True):
        id_rmse = float(group[group["scenario_type"] == "id"]["rmse"].mean())
        ood = group[group["scenario_type"] != "id"]
        ood_rmse = float(ood["rmse"].mean())
        harder_scenarios = int(
            sum(
                float(scenario["rmse"].mean()) > id_rmse + 1e-12
                for _, scenario in ood.groupby("scenario_type")
            )
        )
        rows.append(
            {
                "model": str(model_id),
                "id_rmse": id_rmse,
                "ood_rmse": ood_rmse,
                "ood_minus_id": ood_rmse - id_rmse,
                "harder_ood_scenario_count": harder_scenarios,
                "passed": bool(ood_rmse > id_rmse + 1e-12),
            }
        )
    table = pd.DataFrame(rows)
    return table, bool(table["passed"].any()), bool((table["harder_ood_scenario_count"] > 0).any())


def _split_overlap_count(dataset: dict[str, Any]) -> dict[str, int]:
    calibration = {
        f"{split}_{i:04d}"
        for split, batch in dataset.items()
        if split.startswith("judge_calibration")
        for i in range(batch.n_trajectories)
    }
    test = {
        f"{split}_{i:04d}"
        for split, batch in dataset.items()
        if split.startswith("judge_test")
        for i in range(batch.n_trajectories)
    }
    return {"scenario_overlap_count": len(calibration & test)}


def run_cstr_sanity(config_path: str | Path, output: str | Path) -> dict[str, Any]:
    config = load_calibrated_config(config_path)
    if config["system_id"] != "cstr":
        raise ValueError("CSTR sanity requires system_id='cstr'")
    out_dir = Path(output)
    _ensure_dir(out_dir)
    dataset = generate_calibrated_data(config, out_dir)
    distribution, distribution_checks = _distribution_checks(dataset)
    distribution.to_csv(out_dir / "cstr_distribution_checks.csv", index=False)
    finite, nonconstant = _finite_and_nonconstant(dataset)
    plausible = _physically_plausible(dataset)
    scenario_types = set().union(*(set(batch.scenario_type) for batch in dataset.values()))
    required_scenarios_present = CSTR_REQUIRED_SCENARIO_TYPES <= scenario_types
    support = SupportDistance()
    support.fit(dataset["model_train"])
    models = [make_model(str(model_id), seed=int(config["seed"]) + idx) for idx, model_id in enumerate(config["models"])]
    for model in models:
        model.fit(dataset["model_train"])
    calibration_table = _score_scenarios(config, dataset, "judge_calibration", models, support)
    test_table = _score_scenarios(config, dataset, "judge_test", models, support)
    model_errors, ood_harder_model, materially_harder = _model_error_checks(test_table)
    model_errors.to_csv(out_dir / "cstr_model_error_checks.csv", index=False)
    labels = _label_checks(config, calibration_table, test_table)
    overlap = _split_overlap_count(dataset)
    distribution_passed = bool(
        distribution_checks["cooling_action_shift"] > 1.25
        and distribution_checks["feed_concentration_shift"] > 1.25
        and distribution_checks["feed_temperature_shift"] > 10.0
    )
    no_nan_metrics = bool(
        np.isfinite(model_errors.select_dtypes(include=[float, int]).to_numpy()).all()
        and np.isfinite(calibration_table.select_dtypes(include=[float, int]).to_numpy()).all()
        and np.isfinite(test_table.select_dtypes(include=[float, int]).to_numpy()).all()
    )
    invalid_reasons = []
    weak_reasons = []
    if not finite:
        invalid_reasons.append("CSTR arrays contain non-finite values")
    if not nonconstant:
        invalid_reasons.append("CSTR trajectories are degenerate")
    if not plausible:
        invalid_reasons.append("CSTR states violate physical bounds")
    if not required_scenarios_present:
        invalid_reasons.append("required CSTR scenario types are missing")
    if not distribution_passed:
        invalid_reasons.append("ID/OOD distributions do not differ")
    if not labels["labels_non_degenerate"]:
        invalid_reasons.append("bad RMSE labels are degenerate")
    if overlap["scenario_overlap_count"] != 0:
        invalid_reasons.append("calibration/test scenario IDs overlap")
    if not no_nan_metrics:
        invalid_reasons.append("metrics contain NaNs or infinities")
    if not ood_harder_model:
        weak_reasons.append("OOD mean RMSE is not higher than ID for any model")
    if not materially_harder:
        weak_reasons.append("no OOD scenario is materially harder than ID")
    if invalid_reasons:
        verdict = "INVALID_CSTR_BENCHMARK"
    elif weak_reasons:
        verdict = "WEAK_CSTR_BENCHMARK"
    else:
        verdict = "VALID_CSTR_BENCHMARK"
    data_summary = {
        "system_id": "cstr",
        "finite": finite,
        "nonconstant": nonconstant,
        "physically_plausible": plausible,
        "required_scenarios_present": required_scenarios_present,
        "scenario_types": sorted(scenario_types),
        "distribution_checks": distribution_checks,
        "split_overlap": overlap,
        "no_nan_metrics": no_nan_metrics,
    }
    _write_json(out_dir / "cstr_data_summary.json", data_summary)
    label_payload = {
        **labels,
        "verdict": verdict,
        "invalid_reasons": invalid_reasons,
        "weak_reasons": weak_reasons,
        "model_error_passed": ood_harder_model,
        "materially_harder_ood": materially_harder,
        "distribution_passed": distribution_passed,
        "split_overlap": overlap,
    }
    _write_json(out_dir / "cstr_label_checks.json", label_payload)
    write_cstr_sanity_report(config, data_summary, distribution_checks, model_errors, label_payload, Path("reports/cstr_sanity_report.md"))
    return label_payload


def write_cstr_sanity_report(
    config: dict[str, Any],
    data_summary: dict[str, Any],
    distribution_checks: dict[str, float],
    model_errors: pd.DataFrame,
    labels: dict[str, Any],
    output: Path,
) -> None:
    _ensure_dir(output.parent)
    distribution_rows = pd.DataFrame(
        [
            {"check": "cooling_action_shift", "value": distribution_checks["cooling_action_shift"], "threshold": 1.25, "passed": distribution_checks["cooling_action_shift"] > 1.25},
            {"check": "feed_concentration_shift", "value": distribution_checks["feed_concentration_shift"], "threshold": 1.25, "passed": distribution_checks["feed_concentration_shift"] > 1.25},
            {"check": "feed_temperature_shift", "value": distribution_checks["feed_temperature_shift"], "threshold": 10.0, "passed": distribution_checks["feed_temperature_shift"] > 10.0},
        ]
    )
    label_rows = pd.DataFrame(labels["roles"])
    required_fixes = "none"
    if labels["verdict"] != "VALID_CSTR_BENCHMARK":
        required_fixes = "; ".join(labels.get("invalid_reasons", []) + labels.get("weak_reasons", []))
    text = f"""# CSTR Sanity Report

## Config

experiment_id: {config["experiment_id"]}
seed: {config["seed"]}
horizon: {config["horizon"]}
dt: {config["dt"]}

## CSTR dynamics summary

state: concentration and temperature. action: cooling command. disturbances: feed concentration, feed temperature, and flow-rate/reaction shift proxy.

## Data split summary

Finite: {data_summary["finite"]}
Nonconstant: {data_summary["nonconstant"]}
Physically plausible: {data_summary["physically_plausible"]}
Scenario types: {", ".join(data_summary["scenario_types"])}

## Distribution checks

{_markdown_table(distribution_rows, ["check", "value", "threshold", "passed"])}

## Model error checks

{_markdown_table(model_errors, ["model", "id_rmse", "ood_rmse", "ood_minus_id", "passed"])}

## Label checks

{_markdown_table(label_rows, ["role", "row_count", "bad_count", "bad_rate", "non_degenerate"])}

## Split overlap checks

scenario_overlap_count: {labels["split_overlap"]["scenario_overlap_count"]}

## Verdict

{labels["verdict"]}

## Explanation

Invalid reasons: {labels.get("invalid_reasons", []) or "none"}
Weak reasons: {labels.get("weak_reasons", []) or "none"}

## Required fixes if weak or invalid

{required_fixes}
"""
    output.write_text(text, encoding="utf-8")


def make_multi_system_calibrated_decision_gate(
    twotank_single: str | Path,
    twotank_seed: str | Path,
    twotank_stress: str | Path,
    cstr_sanity: str | Path,
    cstr_single: str | Path,
    cstr_seed: str | Path,
    cstr_stress: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    twotank_single_data = _load_json(Path(twotank_single))
    twotank_seed_data = _load_json(Path(twotank_seed))
    twotank_stress_data = _load_json(Path(twotank_stress))
    cstr_sanity_data = _load_json(Path(cstr_sanity))
    cstr_single_data = _load_json(Path(cstr_single))
    cstr_seed_data = _load_json(Path(cstr_seed))
    cstr_stress_data = _load_json(Path(cstr_stress))
    leakage = bool(
        twotank_single_data.get("leakage_detected")
        or twotank_seed_data.get("leakage_detected")
        or twotank_stress_data.get("leakage_detected")
        or cstr_single_data.get("leakage_detected")
        or cstr_seed_data.get("leakage_detected")
        or cstr_stress_data.get("leakage_detected")
    )
    twotank_robust = (
        twotank_single_data.get("verdict") == "SUPPORTED_LOW_COVERAGE"
        and twotank_seed_data.get("verdict") == "ROBUST_LOW_COVERAGE"
        and twotank_stress_data.get("verdict") == "ROBUST_LOW_COVERAGE_ONLY"
    )
    cstr_sanity_valid = cstr_sanity_data.get("verdict") == "VALID_CSTR_BENCHMARK"
    cstr_robust = (
        cstr_single_data.get("verdict") == "SUPPORTED_LOW_COVERAGE"
        and cstr_seed_data.get("verdict") == "ROBUST_LOW_COVERAGE"
        and cstr_stress_data.get("verdict") == "ROBUST_LOW_COVERAGE_ONLY"
    )
    if leakage:
        decision = "INVALID_DUE_TO_LEAKAGE"
    elif not cstr_sanity_valid:
        decision = "INVALID_CSTR_BENCHMARK"
    elif twotank_robust and cstr_robust:
        decision = "TWO_SYSTEM_LOW_COVERAGE_SUPPORTED"
    elif twotank_robust and cstr_seed_data.get("verdict") == "NO_ROBUST_IMPROVEMENT":
        decision = "TWOTANK_ONLY_SUPPORTED"
    elif (
        cstr_single_data.get("verdict") == "MIXED"
        or cstr_seed_data.get("verdict") == "UNSTABLE"
        or cstr_stress_data.get("verdict") == "THRESHOLD_DEPENDENT"
    ):
        decision = "MIXED_SYSTEM_EVIDENCE"
    elif cstr_sanity_valid:
        decision = "NO_GENERALIZATION"
    else:
        decision = "INVALID_CSTR_BENCHMARK"
    allowed_claims = {
        "TWO_SYSTEM_LOW_COVERAGE_SUPPORTED": "A calibrated low-coverage refusal result replicated on TwoTank and CSTR under the frozen protocol.",
        "TWOTANK_ONLY_SUPPORTED": "The calibrated low-coverage result remains supported on TwoTank only.",
        "MIXED_SYSTEM_EVIDENCE": "Evidence is mixed; no broad generalization claim is supported.",
        "NO_GENERALIZATION": "The CSTR replication failed; no generalization claim is supported.",
        "INVALID_DUE_TO_LEAKAGE": "No claim is allowed because leakage invalidated the evidence.",
        "INVALID_CSTR_BENCHMARK": "No CSTR claim is allowed because benchmark sanity failed.",
    }
    result = {
        "twotank_single_verdict": twotank_single_data.get("verdict"),
        "twotank_seed_verdict": twotank_seed_data.get("verdict"),
        "twotank_stress_verdict": twotank_stress_data.get("verdict"),
        "cstr_sanity_verdict": cstr_sanity_data.get("verdict"),
        "cstr_single_verdict": cstr_single_data.get("verdict"),
        "cstr_seed_verdict": cstr_seed_data.get("verdict"),
        "cstr_stress_verdict": cstr_stress_data.get("verdict"),
        "leakage_detected": leakage,
        "decision": decision,
        "allowed_claim": allowed_claims[decision],
        "forbidden_claims": ["product readiness", "safety certification", "autonomous control", "broad plant-wide validity"],
        "allowed_next_actions": ["write up the bounded evidence", "inspect failure modes", "only then consider separately gated expansion"],
        "forbidden_next_actions": ["RSSM evidence in this milestone", "heat_exchanger evidence in this milestone", "API/frontend/product work"],
    }
    output_path = Path(output)
    _ensure_dir(output_path.parent)
    _write_json(output_path.with_suffix(".json"), result)
    report = f"""# Multi-System Calibrated Decision Gate

## Starting point

TwoTank calibrated low-coverage claim passed. CSTR is tested as frozen-protocol replication.

## Protocol lock

docs/calibrated_protocol_lock_v1.md

## TwoTank evidence

single: {result["twotank_single_verdict"]}
seed: {result["twotank_seed_verdict"]}
stress: {result["twotank_stress_verdict"]}

## CSTR sanity

{result["cstr_sanity_verdict"]}

## CSTR single-run result

{result["cstr_single_verdict"]}

## CSTR seed-sweep result

{result["cstr_seed_verdict"]}

## CSTR threshold/coverage stress result

{result["cstr_stress_verdict"]}

## Leakage status

{leakage}

## Decision

{decision}

## Allowed claims

{result["allowed_claim"]}

## Forbidden claims

{", ".join(result["forbidden_claims"])}

## Allowed next actions

{", ".join(result["allowed_next_actions"])}

## Forbidden next actions

{", ".join(result["forbidden_next_actions"])}

## Explanation

Decision follows the frozen multi-system gate rules. Oracle remains diagnostic only.
"""
    output_path.write_text(report, encoding="utf-8")
    return result
