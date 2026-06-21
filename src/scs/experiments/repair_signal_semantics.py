from __future__ import annotations

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

from scs.data.schemas import load_dataset
from scs.experiments import calibrated as calibrated_exp
from scs.experiments.cstr_weakness import (
    _accepted_mask,
    _auroc,
    _average_precision,
    _distribution,
    _ensure_dir,
    _load_json,
    _markdown_table,
    _scan_forbidden_runtime_refs,
    _spearman,
    _write_json,
)
from scs.experiments.registry import make_model, make_system
from scs.metrics.trajectory import rmse
from scs.validators.calibrated import CALIBRATED_JUDGE_IDS
from scs.validators.invariants import invariant_residual_score
from scs.validators.repair import repair_amount_score
from scs.validators.signal_semantics import write_signal_semantics_artifacts
from scs.validators.support import SupportDistance


BASE_SIGNALS = [
    "support_distance",
    "uncertainty_score",
    "disagreement_score",
    "invariant_residual",
    "repair_amount",
]
SYSTEMS = ["two_tank", "cstr"]
BASELINE_JUDGE = "best_single_signal_selected_on_calibration"
CALIBRATED_REFERENCE_JUDGE = "calibration_selected_candidate_ranker"
DIAGNOSTIC_ORACLE = "oracle_error_rank"
SIMPLE_BY_SIGNAL = {
    "support_distance": "support_only",
    "uncertainty_score": "uncertainty_only",
    "disagreement_score": "disagreement_only",
    "invariant_residual": "invariant_only",
    "repair_amount": "repair_only",
}
CALIBRATED_CONFIG_BY_SYSTEM = {
    "two_tank": Path("configs/experiments/calibrated_two_tank.yaml"),
    "cstr": Path("configs/experiments/calibrated_cstr.yaml"),
}
CALIBRATED_RESULTS_BY_SYSTEM = {
    "two_tank": Path("results/calibrated_two_tank"),
    "cstr": Path("results/calibrated_cstr"),
}
OLD_REPO_NAMES = [
    "time" + "-series" + "-simulator",
    "digital" + "-twin" + "-engine",
    "flux" + "-attention" + "-engine",
    "plant" + "-scenario" + "-compiler",
]


def _read_yaml(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return data


def load_repair_signal_semantics_config(path: str | Path) -> dict[str, Any]:
    config = _read_yaml(path)
    required = {
        "audit_id",
        "systems",
        "source_artifacts",
        "primary_coverages",
        "bad_rmse_threshold",
        "practical_thresholds",
        "baseline_judge",
        "calibrated_reference_judge",
        "diagnostic_oracle",
        "allowed_models",
        "base_signals",
        "signal_sets",
        "diagnostic_thresholds",
        "forbidden",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"missing repair signal semantics config keys: {missing}")
    if list(config["systems"]) != SYSTEMS:
        raise ValueError("repair signal semantics audit must use systems [two_tank, cstr]")
    if list(config["base_signals"]) != BASE_SIGNALS:
        raise ValueError("base_signals must match the existing five deployable signals")
    if config["baseline_judge"] != BASELINE_JUDGE:
        raise ValueError("baseline_judge changed from frozen calibrated protocol")
    if config["calibrated_reference_judge"] != CALIBRATED_REFERENCE_JUDGE:
        raise ValueError("calibrated_reference_judge changed from frozen calibrated protocol")
    if config["diagnostic_oracle"] != DIAGNOSTIC_ORACLE:
        raise ValueError("diagnostic_oracle changed from frozen calibrated protocol")
    if list(config["allowed_models"]) != ["hold_last", "linear_narx", "mlp_state_space"]:
        raise ValueError("allowed_models changed from frozen calibrated protocol")
    for signal_set_id, signals in config["signal_sets"].items():
        unknown = sorted(set(signals) - set(BASE_SIGNALS))
        if unknown:
            raise ValueError(f"signal set {signal_set_id} uses unknown signals: {unknown}")
        if not signals:
            raise ValueError(f"signal set {signal_set_id} is empty")
    expected_sets = {"full_original", "no_repair", "invariant_only", "repair_only", "no_repair_no_uncertainty"}
    if set(config["signal_sets"]) != expected_sets:
        raise ValueError(f"signal_sets must be exactly {sorted(expected_sets)}")
    forbidden = config["forbidden"]
    for key in [
        "allow_new_systems",
        "allow_new_models",
        "allow_new_signals",
        "allow_new_judge_families",
        "allow_protocol_mutation",
        "allow_prior_artifact_overwrite",
    ]:
        if forbidden.get(key) is not False:
            raise ValueError(f"forbidden.{key} must be false")
    thresholds = config["diagnostic_thresholds"]
    for key in [
        "repair_near_zero_epsilon",
        "min_cstr_absolute_margin_improvement",
        "max_allowed_twotank_margin_drop",
        "min_repair_auroc_useful",
        "max_repair_auroc_blind",
    ]:
        if key not in thresholds:
            raise ValueError(f"missing diagnostic threshold: {key}")
    practical = config["practical_thresholds"]
    if float(practical["minimum_absolute_far_reduction"]) != 0.05:
        raise ValueError("minimum_absolute_far_reduction must remain 0.05")
    if float(practical["minimum_relative_far_reduction"]) != 0.10:
        raise ValueError("minimum_relative_far_reduction must remain 0.10")
    return config


def _required_source_artifacts(config: dict[str, Any]) -> list[str]:
    artifacts = [str(value) for value in config["source_artifacts"].values()]
    artifacts.extend(
        [
            "reports/practical_utility_decision_gate.json",
            "reports/cstr_weakness_diagnosis.json",
            "results/calibrated_two_tank/calibration_table.csv",
            "results/calibrated_two_tank/test_table.csv",
            "results/calibrated_cstr/calibration_table.csv",
            "results/calibrated_cstr/test_table.csv",
        ]
    )
    return artifacts


def _git_dirty_lines() -> list[str]:
    if not Path(".git").exists():
        return []
    status = subprocess.run(["git", "status", "--short"], check=True, capture_output=True, text=True)
    return [line for line in status.stdout.splitlines() if line.strip()]


def verify_repair_signal_semantics_preconditions(
    config_path: str | Path,
    output: str | Path,
    report_output: str | Path | None = "reports/repair_signal_semantics_precondition_check.md",
) -> dict[str, Any]:
    config = load_repair_signal_semantics_config(config_path)
    out_dir = Path(output)
    _ensure_dir(out_dir)
    required = _required_source_artifacts(config)
    missing = [path for path in required if not Path(path).exists() or Path(path).stat().st_size == 0]
    practical_gate = _load_json("reports/practical_utility_decision_gate.json")
    cstr_diagnosis = _load_json("reports/cstr_weakness_diagnosis.json")
    scan = _scan_forbidden_runtime_refs([Path("src"), Path("scripts")])
    audit_text = Path(config_path).read_text(encoding="utf-8").lower()
    forbidden_evidence_refs = []
    if "heat_exchanger" in audit_text:
        forbidden_evidence_refs.append("heat_exchanger referenced as evidence")
    if "rssm" in audit_text:
        forbidden_evidence_refs.append("RSSM referenced as evidence")

    decision = practical_gate.get("decision")
    cstr_final = cstr_diagnosis.get("final_diagnosis")
    cstr_next = cstr_diagnosis.get("recommended_next_action")
    expansion_allowed = bool(practical_gate.get("expansion_allowed", True) or cstr_diagnosis.get("expansion_allowed", True))
    reasons: list[str] = []
    if missing:
        reasons.append(f"missing artifacts: {missing}")
    if decision != "NARROW_TO_WEAK_LOW_COVERAGE_CLAIM":
        reasons.append(f"practical utility gate decision is {decision!r}")
    if cstr_final != "REPAIR_SIGNAL_BLIND_SPOT":
        reasons.append(f"CSTR weakness diagnosis is {cstr_final!r}")
    if cstr_next != "FIX_REPAIR_SIGNAL":
        reasons.append(f"CSTR recommended next action is {cstr_next!r}")
    if expansion_allowed:
        reasons.append("expansion is not forbidden")
    if not Path("docs/calibrated_protocol_lock_v1.md").exists():
        reasons.append("protocol lock missing")
    if scan["old_repo_runtime_import_hits"] or scan["path_hack_hits"]:
        reasons.append("forbidden runtime dependency/path scan failed")
    reasons.extend(forbidden_evidence_refs)
    verdict = "READY_FOR_REPAIR_SIGNAL_SEMANTICS_AUDIT" if not reasons else "NOT_READY"
    result = {
        "audit_id": config["audit_id"],
        "working_tree_dirty": bool(_git_dirty_lines()),
        "dirty_state": _git_dirty_lines(),
        "current_controlling_decision": decision,
        "current_cstr_weakness_diagnosis": cstr_final,
        "recommended_next_action": cstr_next,
        "expansion_allowed": expansion_allowed,
        "required_artifacts": [{"path": path, "exists": path not in missing} for path in required],
        "protocol_lock_exists": Path("docs/calibrated_protocol_lock_v1.md").exists(),
        "forbidden_dependency_scan": scan,
        "forbidden_evidence_refs": forbidden_evidence_refs,
        "prior_artifact_mutation_policy": "prior calibrated/effect/CSTR weakness evidence directories are read-only for this audit",
        "verdict": verdict,
        "reasons": reasons,
    }
    _write_json(out_dir / "precondition_check.json", result)
    if report_output is not None:
        write_repair_precondition_report(result, Path(report_output))
    if verdict != "READY_FOR_REPAIR_SIGNAL_SEMANTICS_AUDIT":
        raise RuntimeError(f"repair signal semantics preconditions failed: {reasons}")
    return result


def write_repair_precondition_report(result: dict[str, Any], output: Path) -> None:
    artifacts = pd.DataFrame(result["required_artifacts"])
    scan = result["forbidden_dependency_scan"]
    dirty = "yes" if result["working_tree_dirty"] else "no"
    text = f"""# Repair-Signal Semantics Preconditions

## Current controlling decision

{result["current_controlling_decision"]}

## Current CSTR weakness diagnosis

{result["current_cstr_weakness_diagnosis"]}; recommended next action: {result["recommended_next_action"]}

## Expansion status

Expansion allowed: {result["expansion_allowed"]}

## Working tree status

Dirty: {dirty}

## Required artifacts

{_markdown_table(artifacts, ["path", "exists"])}

## Protocol lock status

Protocol lock exists: {result["protocol_lock_exists"]}

## Forbidden dependency scan

Old repo runtime import hits: {scan["old_repo_runtime_import_hits"] or "none"}
Path hack hits: {scan["path_hack_hits"] or "none"}
Forbidden evidence refs: {result["forbidden_evidence_refs"] or "none"}

## Prior-artifact mutation policy

{result["prior_artifact_mutation_policy"]}

## Verdict

{result["verdict"]}
"""
    _ensure_dir(output.parent)
    output.write_text(text, encoding="utf-8")


def _clip(system: Any, states: np.ndarray) -> np.ndarray:
    if hasattr(system, "clip_trajectory"):
        return system.clip_trajectory(states)
    return np.asarray(states, dtype=float)


def _violates_bounds(system: Any, states: np.ndarray, epsilon: float) -> bool:
    states = np.asarray(states, dtype=float)
    repaired = _clip(system, states)
    return bool(np.max(np.abs(states - repaired)) > epsilon)


def _controlled_case_rows(system_id: str, dt: float, epsilon: float, bad_threshold: float) -> list[dict[str, Any]]:
    system = make_system(system_id)
    has_bounds = hasattr(system, "clip_trajectory")
    has_repair = hasattr(system, "repair_amount")
    if system_id == "two_tank":
        initial = np.array([4.5, 3.0], dtype=float)
        actions = np.full((12, 1), 0.7, dtype=float)
        disturbances = np.column_stack([np.full(12, 0.4), np.full(12, 0.3)])
        true_states = system.rollout(initial, actions, disturbances, dt)
        cases = {
            "valid_in_bounds_trajectory": true_states.copy(),
            "out_of_bounds_negative_inventory": true_states.copy(),
            "out_of_bounds_over_capacity": true_states.copy(),
            "within_bounds_wrong_dynamics": np.clip(true_states + np.array([2.0, -1.5]), 0.1, 9.5),
        }
        cases["out_of_bounds_negative_inventory"][4, 0] = -1.25
        cases["out_of_bounds_over_capacity"][5, 1] = 12.0
    else:
        initial = np.array([1.0, 340.0], dtype=float)
        actions = np.full((12, 1), 5.0, dtype=float)
        disturbances = np.column_stack([np.full(12, 1.1), np.full(12, 335.0), np.full(12, 0.35)])
        true_states = system.rollout(initial, actions, disturbances, dt)
        cases = {
            "valid_in_bounds_trajectory": true_states.copy(),
            "out_of_bounds_temperature": true_states.copy(),
            "out_of_bounds_concentration": true_states.copy(),
            "within_bounds_wrong_reaction_dynamics": true_states.copy(),
            "within_bounds_wrong_temperature_trajectory": true_states.copy(),
        }
        cases["out_of_bounds_temperature"][3, 1] = 530.0
        cases["out_of_bounds_concentration"][4, 0] = 2.4
        cases["within_bounds_wrong_reaction_dynamics"][:, 0] = np.clip(true_states[:, 0] + 0.55, 0.05, 1.95)
        cases["within_bounds_wrong_temperature_trajectory"][:, 1] = np.clip(true_states[:, 1] + 35.0, 255.0, 495.0)

    expected_positive = {
        "out_of_bounds_negative_inventory",
        "out_of_bounds_over_capacity",
        "out_of_bounds_temperature",
        "out_of_bounds_concentration",
    }
    expected_near_zero = {
        "valid_in_bounds_trajectory",
        "within_bounds_wrong_dynamics",
        "within_bounds_wrong_reaction_dynamics",
        "within_bounds_wrong_temperature_trajectory",
    }
    rows = []
    for case_id, predicted in cases.items():
        repair = repair_amount_score(system, predicted) if has_repair else 0.0
        inv = float(np.mean(invariant_residual_score(system, predicted, actions, disturbances, dt)))
        case_rmse = float(rmse(predicted, true_states))
        violates = _violates_bounds(system, predicted, epsilon)
        wants_positive = case_id in expected_positive
        wants_near_zero = case_id in expected_near_zero
        passed = True
        if wants_positive and has_bounds and has_repair:
            passed = passed and repair > epsilon
        if wants_near_zero:
            passed = passed and repair <= epsilon
        if case_id.startswith("within_bounds_wrong"):
            passed = passed and case_rmse > bad_threshold
        rows.append(
            {
                "system_id": system_id,
                "case_id": case_id,
                "has_bounds": bool(has_bounds),
                "has_repair_operator": bool(has_repair),
                "raw_state_violates_bounds": bool(violates),
                "repair_amount": float(repair),
                "expected_repair_positive": bool(wants_positive),
                "expected_repair_near_zero": bool(wants_near_zero),
                "rmse": case_rmse,
                "invariant_residual": inv,
                "case_passed": bool(passed),
                "interpretation": _controlled_case_interpretation(case_id, repair, case_rmse, epsilon),
            }
        )
    return rows


def _controlled_case_interpretation(case_id: str, repair: float, case_rmse: float, epsilon: float) -> str:
    if case_id.startswith("out_of_bounds"):
        return "bounds violation should produce positive repair"
    if case_id.startswith("within_bounds_wrong"):
        return "within-bound wrong dynamics should have high RMSE but near-zero repair"
    if repair <= epsilon and case_rmse <= epsilon:
        return "valid trajectory needs no repair"
    return "controlled repair case"


def repair_validation_verdict(cases: pd.DataFrame, epsilon: float, bad_threshold: float) -> str:
    defined = cases[cases["has_bounds"] & cases["has_repair_operator"]]
    cstr = cases[cases["system_id"] == "cstr"]
    if cstr.empty or not bool(cstr["has_repair_operator"].any()) or not bool(cstr["has_bounds"].any()):
        return "CSTR_REPAIR_NOT_DEFINED"
    positive_cases = defined[defined["expected_repair_positive"]]
    if not positive_cases.empty and (positive_cases["repair_amount"].astype(float) <= epsilon).any():
        return "REPAIR_IMPLEMENTATION_BUG"
    wrong = cstr[cstr["case_id"].astype(str).str.startswith("within_bounds_wrong")]
    if (
        not wrong.empty
        and (wrong["repair_amount"].astype(float) <= epsilon).all()
        and (wrong["rmse"].astype(float) > bad_threshold).all()
    ):
        return "REPAIR_CORRECT_BUT_CSTR_IRRELEVANT"
    return "INCONCLUSIVE"


def validate_repair_amount_semantics(
    config_path: str | Path,
    output: str | Path,
    report_output: str | Path | None = "reports/repair_amount_semantics_validation.md",
) -> dict[str, Any]:
    config = load_repair_signal_semantics_config(config_path)
    out_dir = Path(output)
    _ensure_dir(out_dir)
    epsilon = float(config["diagnostic_thresholds"]["repair_near_zero_epsilon"])
    threshold = float(config["bad_rmse_threshold"])
    rows = []
    for system_id in config["systems"]:
        rows.extend(_controlled_case_rows(str(system_id), dt=0.1, epsilon=epsilon, bad_threshold=threshold))
    cases = pd.DataFrame(rows)
    cases.to_csv(out_dir / "controlled_repair_cases.csv", index=False)
    verdict = repair_validation_verdict(cases, epsilon, threshold)
    summary = {
        "verdict": verdict,
        "epsilon": epsilon,
        "bad_rmse_threshold": threshold,
        "two_tank_repair_status": _repair_status_for_system(cases, "two_tank", epsilon),
        "cstr_repair_status": _repair_status_for_system(cases, "cstr", epsilon),
        "cstr_repair_semantic_status": "within-bound CSTR errors are high-RMSE but near-zero repair"
        if verdict == "REPAIR_CORRECT_BUT_CSTR_IRRELEVANT"
        else verdict,
        "failed_cases": cases.loc[~cases["case_passed"], ["system_id", "case_id"]].to_dict(orient="records"),
    }
    _write_json(out_dir / "repair_validation_summary.json", summary)
    if report_output is not None:
        write_repair_validation_report(cases, summary, Path(report_output))
    failed_positive = cases[cases["expected_repair_positive"] & ~cases["case_passed"]]
    if not failed_positive.empty:
        raise RuntimeError(f"repair-positive controlled cases failed: {failed_positive[['system_id', 'case_id']].to_dict(orient='records')}")
    return summary


def _repair_status_for_system(cases: pd.DataFrame, system_id: str, epsilon: float) -> str:
    system_cases = cases[cases["system_id"] == system_id]
    positive = system_cases[system_cases["expected_repair_positive"]]
    wrong = system_cases[system_cases["case_id"].astype(str).str.startswith("within_bounds_wrong")]
    if system_cases.empty or not bool(system_cases["has_repair_operator"].any()):
        return "REPAIR_NOT_DEFINED"
    if not positive.empty and (positive["repair_amount"].astype(float) > epsilon).all() and bool(positive["case_passed"].all()):
        if not wrong.empty and (wrong["repair_amount"].astype(float) <= epsilon).all():
            return "REPAIR_BOUNDS_ONLY"
        return "REPAIR_WORKS_ON_BOUNDS"
    return "REPAIR_FAILED_CONTROLLED_CASE"


def write_repair_validation_report(cases: pd.DataFrame, summary: dict[str, Any], output: Path) -> None:
    table = cases.rename(columns={"system_id": "system", "case_id": "case", "raw_state_violates_bounds": "violates_bounds"})
    table["expected"] = np.where(table["expected_repair_positive"], "positive", "near_zero")
    text = f"""# Repair Amount Semantics Validation

## Question

Is repair_amount broken or semantically irrelevant for CSTR?

## Controlled cases

{_markdown_table(table, ["system", "case", "violates_bounds", "repair_amount", "expected", "case_passed"])}

## CSTR repair status

{summary["cstr_repair_status"]}

## TwoTank repair status

{summary["two_tank_repair_status"]}

## Interpretation

{summary["cstr_repair_semantic_status"]}

## Verdict

{summary["verdict"]}
"""
    _ensure_dir(output.parent)
    output.write_text(text, encoding="utf-8")


def _system_table(system_id: str, kind: str) -> pd.DataFrame:
    return pd.read_csv(CALIBRATED_RESULTS_BY_SYSTEM[system_id] / f"{kind}_table.csv")


def _reference_accepted_frame(table: pd.DataFrame, coverage: float, reference_judge: str) -> pd.DataFrame:
    risk_column = f"risk_{reference_judge}"
    rows = []
    for (_, _), group in table.groupby(["model_id", "scenario_type"], sort=False):
        group = group.copy()
        group["accepted"] = _accepted_mask(group, risk_column, coverage).to_numpy(dtype=bool)
        rows.append(group)
    frame = pd.concat(rows, ignore_index=True)
    frame["accepted_region"] = np.where(
        frame["accepted"] & frame["bad_rmse_label"].astype(bool),
        "accepted_bad",
        np.where(
            frame["accepted"],
            "accepted_good",
            np.where(frame["bad_rmse_label"].astype(bool), "rejected_bad", "rejected_good"),
        ),
    )
    return frame


def _signal_low_threshold(frame: pd.DataFrame, signal: str, epsilon: float) -> float:
    if signal == "repair_amount":
        return epsilon
    return float(frame[signal].quantile(0.25))


def compare_repair_vs_invariant(
    config_path: str | Path,
    output: str | Path,
    report_output: str | Path | None = "reports/repair_vs_invariant_comparison.md",
) -> dict[str, Any]:
    config = load_repair_signal_semantics_config(config_path)
    out_dir = Path(output)
    _ensure_dir(out_dir)
    epsilon = float(config["diagnostic_thresholds"]["repair_near_zero_epsilon"])
    reference_judge = str(config["calibrated_reference_judge"])
    metrics_rows: list[dict[str, Any]] = []
    joint_rows: list[dict[str, Any]] = []
    for system_id in config["systems"]:
        table = _system_table(str(system_id), "test")
        table["bad_rmse_label"] = table["rmse"].astype(float) > float(config["bad_rmse_threshold"])
        for coverage in [float(value) for value in config["primary_coverages"]]:
            accepted = _reference_accepted_frame(table, coverage, reference_judge)
            high_invariant_threshold = float(accepted["invariant_residual"].quantile(0.75))
            low_invariant_threshold = float(accepted["invariant_residual"].quantile(0.25))
            bad_accepted = accepted[accepted["accepted_region"] == "accepted_bad"]
            joint_rows.append(
                {
                    "system_id": system_id,
                    "coverage": coverage,
                    "low_repair_bad_count": int(np.sum(bad_accepted["repair_amount"].astype(float) <= epsilon)),
                    "low_repair_high_invariant_bad_count": int(
                        np.sum(
                            (bad_accepted["repair_amount"].astype(float) <= epsilon)
                            & (bad_accepted["invariant_residual"].astype(float) > high_invariant_threshold)
                        )
                    ),
                    "low_repair_low_invariant_bad_count": int(
                        np.sum(
                            (bad_accepted["repair_amount"].astype(float) <= epsilon)
                            & (bad_accepted["invariant_residual"].astype(float) <= low_invariant_threshold)
                        )
                    ),
                    "high_repair_bad_count": int(np.sum(bad_accepted["repair_amount"].astype(float) > epsilon)),
                    "high_invariant_threshold": high_invariant_threshold,
                    "low_invariant_threshold": low_invariant_threshold,
                }
            )
            for signal in ["repair_amount", "invariant_residual"]:
                low_threshold = _signal_low_threshold(accepted, signal, epsilon)
                global_metrics = _signal_metrics_for_frame(accepted, signal, "all", low_threshold)
                global_metrics.update({"system_id": system_id, "coverage": coverage})
                metrics_rows.append(global_metrics)
                for region, group in accepted.groupby("accepted_region", sort=True):
                    region_metrics = _signal_metrics_for_frame(group, signal, str(region), low_threshold)
                    region_metrics.update({"system_id": system_id, "coverage": coverage})
                    metrics_rows.append(region_metrics)
            _plot_repair_invariant_scatter(
                accepted,
                out_dir / f"repair_vs_invariant_scatter_{'twotank' if system_id == 'two_tank' else 'cstr'}.png",
                f"{system_id} repair vs invariant at coverage {coverage}",
            )
    metrics = pd.DataFrame(metrics_rows)
    joint = pd.DataFrame(joint_rows)
    metrics.to_csv(out_dir / "repair_vs_invariant_metrics.csv", index=False)
    joint.to_csv(out_dir / "joint_signal_cases.csv", index=False)
    summary = repair_vs_invariant_summary(metrics, joint, config)
    _write_json(out_dir / "repair_vs_invariant_summary.json", summary)
    if report_output is not None:
        write_repair_vs_invariant_report(metrics, joint, summary, Path(report_output))
    return summary


def _signal_metrics_for_frame(frame: pd.DataFrame, signal: str, group: str, low_threshold: float) -> dict[str, Any]:
    dist = _distribution(frame[signal])
    labels = frame["bad_rmse_label"].astype(bool)
    values = frame[signal].astype(float)
    accepted_bad = frame[frame["accepted_region"] == "accepted_bad"] if "accepted_region" in frame else frame.iloc[0:0]
    if len(frame) and labels.nunique() > 1 and values.nunique() > 1:
        auc = _auroc(labels, values)
        auprc = _average_precision(labels, values)
    else:
        auc = 0.5
        auprc = float(labels.mean()) if len(labels) else 0.0
    return {
        "signal": signal,
        "accepted_region": group,
        "mean": dist["mean"],
        "median": dist["median"],
        "p10": dist["p10"],
        "p90": dist["p90"],
        "auroc_for_bad_rmse_label": 0.5 if auc is None else float(auc),
        "auprc_for_bad_rmse_label": 0.0 if auprc is None else float(auprc),
        "spearman_correlation_with_rmse": 0.0 if _spearman(values, frame["rmse"]) is None else float(_spearman(values, frame["rmse"])),
        "near_zero_fraction": float(np.mean(np.isclose(values, 0.0))) if len(values) else 0.0,
        "accepted_bad_low_signal_fraction": float(np.mean(accepted_bad[signal].astype(float) <= low_threshold)) if len(accepted_bad) else 0.0,
        "low_signal_threshold": float(low_threshold),
    }


def _plot_repair_invariant_scatter(frame: pd.DataFrame, output: Path, title: str) -> None:
    _ensure_dir(output.parent)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    colors = np.where(frame["bad_rmse_label"].astype(bool), "tab:red", "tab:blue")
    ax.scatter(frame["repair_amount"].astype(float), frame["invariant_residual"].astype(float), c=colors, s=12, alpha=0.45)
    ax.set_xlabel("repair_amount")
    ax.set_ylabel("invariant_residual")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def repair_vs_invariant_summary(metrics: pd.DataFrame, joint: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    global_rows = metrics[metrics["accepted_region"] == "all"]
    avg = (
        global_rows.groupby(["system_id", "signal"], as_index=False)
        .agg(
            auroc=("auroc_for_bad_rmse_label", "mean"),
            auprc=("auprc_for_bad_rmse_label", "mean"),
            spearman_rmse=("spearman_correlation_with_rmse", "mean"),
            near_zero_fraction=("near_zero_fraction", "mean"),
        )
    )
    def _val(system_id: str, signal: str, column: str) -> float:
        row = avg[(avg["system_id"] == system_id) & (avg["signal"] == signal)]
        return 0.0 if row.empty else float(row.iloc[0][column])

    cstr_repair_auc = _val("cstr", "repair_amount", "auroc")
    cstr_invariant_auc = _val("cstr", "invariant_residual", "auroc")
    twotank_repair_auc = _val("two_tank", "repair_amount", "auroc")
    twotank_invariant_auc = _val("two_tank", "invariant_residual", "auroc")
    max_blind = float(config["diagnostic_thresholds"]["max_repair_auroc_blind"])
    min_useful = float(config["diagnostic_thresholds"]["min_repair_auroc_useful"])
    if cstr_invariant_auc - cstr_repair_auc >= 0.20 and cstr_repair_auc < max_blind:
        verdict = "INVARIANT_DOMINATES_REPAIR"
    elif twotank_repair_auc >= min_useful and cstr_repair_auc < max_blind:
        verdict = "REPAIR_SYSTEM_SPECIFIC"
    elif min(cstr_repair_auc, cstr_invariant_auc, twotank_repair_auc, twotank_invariant_auc) >= min_useful:
        verdict = "BOTH_USEFUL"
    elif max(cstr_repair_auc, cstr_invariant_auc, twotank_repair_auc, twotank_invariant_auc) < max_blind:
        verdict = "BOTH_WEAK"
    else:
        verdict = "INCONCLUSIVE"
    return {
        "verdict": verdict,
        "cstr_repair_auroc": cstr_repair_auc,
        "cstr_invariant_auroc": cstr_invariant_auc,
        "two_tank_repair_auroc": twotank_repair_auc,
        "two_tank_invariant_auroc": twotank_invariant_auc,
        "low_repair_high_invariant_bad_count_total": int(joint["low_repair_high_invariant_bad_count"].sum()) if not joint.empty else 0,
    }


def write_repair_vs_invariant_report(metrics: pd.DataFrame, joint: pd.DataFrame, summary: dict[str, Any], output: Path) -> None:
    performance = (
        metrics[metrics["accepted_region"] == "all"]
        .groupby(["system_id", "signal"], as_index=False)
        .agg(
            auroc=("auroc_for_bad_rmse_label", "mean"),
            auprc=("auprc_for_bad_rmse_label", "mean"),
            spearman_rmse=("spearman_correlation_with_rmse", "mean"),
            near_zero_fraction=("near_zero_fraction", "mean"),
        )
        .rename(columns={"system_id": "system"})
    )
    accepted = (
        joint.groupby("system_id", as_index=False)
        .agg(
            low_repair_bad=("low_repair_bad_count", "sum"),
            low_repair_high_invariant_bad=("low_repair_high_invariant_bad_count", "sum"),
            low_repair_low_invariant_bad=("low_repair_low_invariant_bad_count", "sum"),
        )
        .rename(columns={"system_id": "system"})
    )
    text = f"""# Repair vs Invariant Residual Comparison

## Question

Is repair_amount less informative than invariant_residual, especially on CSTR?

## Signal performance by system

{_markdown_table(performance, ["system", "signal", "auroc", "auprc", "spearman_rmse", "near_zero_fraction"])}

## Accepted false accepts

{_markdown_table(accepted, ["system", "low_repair_bad", "low_repair_high_invariant_bad", "low_repair_low_invariant_bad"])}

## Interpretation

CSTR repair AUROC={summary["cstr_repair_auroc"]:.6f}; CSTR invariant AUROC={summary["cstr_invariant_auroc"]:.6f}.

## Verdict

{summary["verdict"]}
"""
    _ensure_dir(output.parent)
    output.write_text(text, encoding="utf-8")


def _signal_set_judges(signals: list[str]) -> list[str]:
    simple = [judge for signal, judge in SIMPLE_BY_SIGNAL.items() if signal in signals]
    return [
        *simple,
        "combined_linear",
        *CALIBRATED_JUDGE_IDS,
        "random_baseline",
        "oracle_error_rank",
    ]


def _base_eval_config(system_id: str, config: dict[str, Any], seed: int, signals: list[str]) -> dict[str, Any]:
    base = calibrated_exp.load_calibrated_config(CALIBRATED_CONFIG_BY_SYSTEM[system_id])
    base = dict(base)
    base["seed"] = int(seed)
    base["signals"] = list(signals)
    base["judges"] = _signal_set_judges(signals)
    base["coverages"] = [float(value) for value in config["primary_coverages"]]
    base["primary_coverages"] = [float(value) for value in config["primary_coverages"]]
    base["bad_threshold"] = {"metric": "rmse", "value": float(config["bad_rmse_threshold"])}
    base["models"] = list(config["allowed_models"])
    return base


def _evaluate_signal_set_on_tables(
    audit_config: dict[str, Any],
    system_id: str,
    signal_set_id: str,
    signals: list[str],
    seed: int,
    calibration_table: pd.DataFrame,
    test_table: pd.DataFrame,
    output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    eval_config = _base_eval_config(system_id, audit_config, seed, signals)
    run_dir = output / f"seed_{seed}" / system_id / signal_set_id
    summary = calibrated_exp.evaluate_calibrated_tables(
        eval_config,
        calibration_table.copy(),
        test_table.copy(),
        run_dir,
        report_path=None,
        command="repair signal semantics audit signal-set evaluation",
    )
    risk = pd.read_csv(run_dir / "calibrated_risk_coverage.csv")
    low = pd.read_csv(run_dir / "low_coverage_summary.csv")
    selection = pd.read_csv(run_dir / "calibration_selection.csv")
    low_rows = []
    for _, row in low.iterrows():
        selected_judge = str(row["best_calibrated_judge"])
        selected = selection[selection["judge_id"] == selected_judge]
        selected_signal = "none" if selected.empty else str(selected.iloc[0]["selected_signal_if_any"])
        selected_risk = risk[
            (risk["judge_id"] == selected_judge)
            & np.isclose(risk["coverage_requested"].astype(float), float(row["coverage"]))
        ]
        accepted_count = int(selected_risk["accepted_count"].sum()) if not selected_risk.empty else 0
        false_accept_count = int(selected_risk["false_accept_count"].sum()) if not selected_risk.empty else 0
        baseline_far = float(row["baseline_far"])
        calibrated_far = float(row["calibrated_far"])
        low_rows.append(
            {
                "system_id": system_id,
                "signal_set_id": signal_set_id,
                "coverage": float(row["coverage"]),
                "baseline_far": baseline_far,
                "calibrated_far": calibrated_far,
                "absolute_margin": baseline_far - calibrated_far,
                "relative_margin": (baseline_far - calibrated_far) / baseline_far if baseline_far > 0 else 0.0,
                "accepted_count": accepted_count,
                "false_accept_count": false_accept_count,
                "selected_judge": selected_judge,
                "selected_signal_if_any": selected_signal,
                "seed": int(seed),
                "leakage_detected": bool(summary.get("leakage_detected", False)),
            }
        )
    low_frame = pd.DataFrame(low_rows)
    risk = risk.assign(system_id=system_id, signal_set_id=signal_set_id, seed=int(seed))
    return low_frame, risk, summary


def _existing_tables_for_system(system_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = CALIBRATED_RESULTS_BY_SYSTEM[system_id]
    return pd.read_csv(root / "calibration_table.csv"), pd.read_csv(root / "test_table.csv")


def _fresh_tables_for_seed(system_id: str, seed: int, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = calibrated_exp.load_calibrated_config(CALIBRATED_CONFIG_BY_SYSTEM[system_id])
    config = dict(config)
    config["seed"] = int(seed)
    config["output_dir"] = str(output / f"seed_{seed}" / system_id / "scored_data")
    dataset = calibrated_exp.generate_calibrated_data(config, Path(config["output_dir"]))
    support = SupportDistance()
    support.fit(dataset["model_train"])
    models = [make_model(str(model_id), seed=int(seed) + idx) for idx, model_id in enumerate(config["models"])]
    for model in models:
        model.fit(dataset["model_train"])
    calibration_table = calibrated_exp._score_scenarios(config, dataset, "judge_calibration", models, support)
    test_table = calibrated_exp._score_scenarios(config, dataset, "judge_test", models, support)
    return calibration_table, test_table


def run_signal_set_ablation(
    config_path: str | Path,
    output: str | Path,
    report_output: str | Path | None = "reports/signal_set_ablation_repair.md",
) -> dict[str, Any]:
    config = load_repair_signal_semantics_config(config_path)
    out_dir = Path(output)
    _ensure_dir(out_dir)
    low_frames = []
    risk_frames = []
    for system_id in config["systems"]:
        calibration_table, test_table = _existing_tables_for_system(str(system_id))
        for signal_set_id, signals in config["signal_sets"].items():
            low, risk, _ = _evaluate_signal_set_on_tables(
                config,
                str(system_id),
                str(signal_set_id),
                list(signals),
                seed=42,
                calibration_table=calibration_table,
                test_table=test_table,
                output=out_dir,
            )
            low_frames.append(low)
            risk_frames.append(risk)
    low_summary = pd.concat(low_frames, ignore_index=True)
    risk_coverage = pd.concat(risk_frames, ignore_index=True)
    seed_summary = low_summary.copy()
    diff = _signal_set_difference_vs_full(low_summary)
    summary = signal_set_ablation_summary(low_summary, diff, config)
    risk_coverage.to_csv(out_dir / "signal_set_risk_coverage.csv", index=False)
    low_summary.to_csv(out_dir / "signal_set_low_coverage_summary.csv", index=False)
    seed_summary.to_csv(out_dir / "signal_set_seed_summary.csv", index=False)
    _write_json(out_dir / "signal_set_ablation_summary.json", summary)
    _plot_signal_set_ablation(low_summary, out_dir / "signal_set_ablation_plot.png")
    if report_output is not None:
        write_signal_set_ablation_report(low_summary, diff, summary, Path(report_output))
    return summary


def _signal_set_difference_vs_full(low: pd.DataFrame) -> pd.DataFrame:
    full = low[low["signal_set_id"] == "full_original"][
        ["system_id", "coverage", "seed", "absolute_margin"]
    ].rename(columns={"absolute_margin": "full_absolute_margin"})
    merged = low.merge(full, on=["system_id", "coverage", "seed"], how="left")
    merged["delta_margin_vs_full"] = merged["absolute_margin"] - merged["full_absolute_margin"]
    return merged[["system_id", "signal_set_id", "coverage", "seed", "delta_margin_vs_full"]]


def signal_set_ablation_summary(low: pd.DataFrame, diff: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    cstr_delta = _mean_delta(diff, "cstr", "no_repair")
    twotank_delta = _mean_delta(diff, "two_tank", "no_repair")
    min_cstr = float(config["diagnostic_thresholds"]["min_cstr_absolute_margin_improvement"])
    max_drop = float(config["diagnostic_thresholds"]["max_allowed_twotank_margin_drop"])
    cstr_improves = cstr_delta >= min_cstr
    twotank_drop = max(0.0, -twotank_delta)
    if cstr_improves and twotank_drop <= max_drop:
        verdict = "NO_REPAIR_IMPROVES_CSTR_WITHOUT_HURTING_TWOTANK"
    elif cstr_improves and twotank_drop > max_drop:
        verdict = "REPAIR_USEFUL_FOR_TWOTANK_ONLY"
    elif not cstr_improves:
        verdict = "NO_REPAIR_NO_BENEFIT"
    else:
        verdict = "SIGNAL_SET_EFFECT_INCONCLUSIVE"
    return {
        "verdict": verdict,
        "cstr_no_repair_mean_delta_margin_vs_full": cstr_delta,
        "two_tank_no_repair_mean_delta_margin_vs_full": twotank_delta,
        "configured_min_cstr_absolute_margin_improvement": min_cstr,
        "configured_max_allowed_twotank_margin_drop": max_drop,
        "leakage_detected": bool(low["leakage_detected"].any()),
    }


def _mean_delta(diff: pd.DataFrame, system_id: str, signal_set_id: str) -> float:
    rows = diff[(diff["system_id"] == system_id) & (diff["signal_set_id"] == signal_set_id)]
    return 0.0 if rows.empty else float(rows["delta_margin_vs_full"].mean())


def _plot_signal_set_ablation(low: pd.DataFrame, output: Path) -> None:
    _ensure_dir(output.parent)
    plot = low.groupby(["system_id", "signal_set_id", "coverage"], as_index=False)["absolute_margin"].mean()
    fig, ax = plt.subplots(figsize=(9, 5))
    for (system_id, signal_set), group in plot.groupby(["system_id", "signal_set_id"], sort=True):
        label = f"{system_id}:{signal_set}"
        ax.plot(group["coverage"], group["absolute_margin"], marker="o", linewidth=1.2, label=label)
    ax.set_xlabel("coverage")
    ax.set_ylabel("absolute_margin")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def write_signal_set_ablation_report(low: pd.DataFrame, diff: pd.DataFrame, summary: dict[str, Any], output: Path) -> None:
    report_low = low.rename(columns={"system_id": "system", "signal_set_id": "signal_set", "absolute_margin": "margin"})
    report_diff = diff.rename(columns={"system_id": "system", "signal_set_id": "signal_set"})
    selected = low[["system_id", "signal_set_id", "coverage", "selected_judge", "selected_signal_if_any"]].drop_duplicates()
    leakage = low.groupby(["system_id", "signal_set_id"], as_index=False)["leakage_detected"].any()
    text = f"""# Signal-Set Ablation: Full vs No Repair

## Question

Does excluding repair_amount improve CSTR without hurting TwoTank?

## Low-coverage result

{_markdown_table(report_low, ["system", "signal_set", "coverage", "baseline_far", "calibrated_far", "margin"])}

## Difference vs full_original

{_markdown_table(report_diff, ["system", "signal_set", "coverage", "delta_margin_vs_full"])}

## Selected signals/judges

{_markdown_table(selected, ["system_id", "signal_set_id", "coverage", "selected_judge", "selected_signal_if_any"])}

## Leakage status

{_markdown_table(leakage, ["system_id", "signal_set_id", "leakage_detected"])}

## Interpretation

CSTR no-repair delta={summary["cstr_no_repair_mean_delta_margin_vs_full"]:.6f}; TwoTank no-repair delta={summary["two_tank_no_repair_mean_delta_margin_vs_full"]:.6f}.

## Verdict

{summary["verdict"]}
"""
    _ensure_dir(output.parent)
    output.write_text(text, encoding="utf-8")


def run_signal_set_ablation_seed_sweep(
    config_path: str | Path,
    seeds: list[int],
    output: str | Path,
    report_output: str | Path | None = "reports/signal_set_ablation_seed_sweep.md",
) -> dict[str, Any]:
    config = load_repair_signal_semantics_config(config_path)
    if len(seeds) < 1:
        raise ValueError("at least one seed is required")
    out_dir = Path(output)
    _ensure_dir(out_dir)
    rows = []
    failures = []
    for seed in seeds:
        for system_id in config["systems"]:
            try:
                calibration_table, test_table = _fresh_tables_for_seed(str(system_id), int(seed), out_dir)
                for signal_set_id, signals in config["signal_sets"].items():
                    low, _, _ = _evaluate_signal_set_on_tables(
                        config,
                        str(system_id),
                        str(signal_set_id),
                        list(signals),
                        int(seed),
                        calibration_table,
                        test_table,
                        out_dir,
                    )
                    low["run_status"] = "passed"
                    rows.append(low)
            except Exception as exc:  # pragma: no cover - exercised through failure artifacts
                failures.append({"seed": int(seed), "system_id": str(system_id), "error": str(exc)})
                rows.append(
                    pd.DataFrame(
                        [
                            {
                                "system_id": str(system_id),
                                "signal_set_id": "all",
                                "coverage": np.nan,
                                "baseline_far": np.nan,
                                "calibrated_far": np.nan,
                                "absolute_margin": np.nan,
                                "relative_margin": np.nan,
                                "accepted_count": 0,
                                "false_accept_count": 0,
                                "selected_judge": "none",
                                "selected_signal_if_any": "none",
                                "seed": int(seed),
                                "leakage_detected": False,
                                "run_status": "failed",
                            }
                        ]
                    )
                )
    results = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    results.to_csv(out_dir / "signal_set_seed_results.csv", index=False)
    summary = signal_set_seed_sweep_summary(results, failures, config, seeds)
    _write_json(out_dir / "signal_set_seed_summary.json", summary)
    if report_output is not None:
        write_signal_set_seed_sweep_report(results, summary, Path(report_output))
    if failures:
        raise RuntimeError(f"signal set seed sweep failures: {failures}")
    return summary


def signal_set_seed_sweep_summary(
    results: pd.DataFrame,
    failures: list[dict[str, Any]],
    config: dict[str, Any],
    seeds: list[int],
) -> dict[str, Any]:
    valid = results[results["run_status"] == "passed"].copy()
    diff = _signal_set_difference_vs_full(valid) if not valid.empty else pd.DataFrame()
    cstr = diff[(diff["system_id"] == "cstr") & (diff["signal_set_id"] == "no_repair")]
    twotank = diff[(diff["system_id"] == "two_tank") & (diff["signal_set_id"] == "no_repair")]
    min_cstr = float(config["diagnostic_thresholds"]["min_cstr_absolute_margin_improvement"])
    max_drop = float(config["diagnostic_thresholds"]["max_allowed_twotank_margin_drop"])
    cstr_seed = cstr.groupby("seed", as_index=False)["delta_margin_vs_full"].mean() if not cstr.empty else pd.DataFrame(columns=["seed", "delta_margin_vs_full"])
    twotank_seed = twotank.groupby("seed", as_index=False)["delta_margin_vs_full"].mean() if not twotank.empty else pd.DataFrame(columns=["seed", "delta_margin_vs_full"])
    cstr_improve_count = int(np.sum(cstr_seed["delta_margin_vs_full"].astype(float) >= min_cstr)) if not cstr_seed.empty else 0
    twotank_harm_count = int(np.sum(twotank_seed["delta_margin_vs_full"].astype(float) < -max_drop)) if not twotank_seed.empty else 0
    if cstr_improve_count >= 7 and twotank_harm_count <= 3:
        verdict = "SEED_STABLE_NO_REPAIR_BENEFIT"
    elif cstr_improve_count > 3 and twotank_harm_count > 3:
        verdict = "HURTS_TWOTANK"
    elif 4 <= cstr_improve_count <= 6:
        verdict = "CSTR_ONLY_UNSTABLE_BENEFIT"
    else:
        verdict = "NO_SEED_STABLE_BENEFIT"
    return {
        "verdict": verdict,
        "seeds": [int(seed) for seed in seeds],
        "failed_seeds": failures,
        "cstr_improve_seed_count": cstr_improve_count,
        "twotank_harm_seed_count": twotank_harm_count,
        "cstr_win_rate": cstr_improve_count / max(len(seeds), 1),
        "twotank_harm_rate": twotank_harm_count / max(len(seeds), 1),
        "cstr_mean_delta_margin": float(cstr_seed["delta_margin_vs_full"].mean()) if not cstr_seed.empty else 0.0,
        "twotank_mean_delta_margin": float(twotank_seed["delta_margin_vs_full"].mean()) if not twotank_seed.empty else 0.0,
    }


def write_signal_set_seed_sweep_report(results: pd.DataFrame, summary: dict[str, Any], output: Path) -> None:
    valid = results[results["run_status"] == "passed"].copy()
    diff = _signal_set_difference_vs_full(valid) if not valid.empty else pd.DataFrame(columns=["system_id", "signal_set_id", "coverage", "seed", "delta_margin_vs_full"])
    cstr = diff[(diff["system_id"] == "cstr") & (diff["signal_set_id"] == "no_repair")]
    twotank = diff[(diff["system_id"] == "two_tank") & (diff["signal_set_id"] == "no_repair")]
    cstr_report = (
        cstr.assign(win=cstr["delta_margin_vs_full"] > 0.0)
        .groupby("coverage", as_index=False)
        .agg(win_rate=("win", "mean"), mean_delta_margin=("delta_margin_vs_full", "mean"), std_delta_margin=("delta_margin_vs_full", "std"))
        if not cstr.empty
        else pd.DataFrame(columns=["coverage", "win_rate", "mean_delta_margin", "std_delta_margin"])
    )
    twotank_report = (
        twotank.assign(harm=twotank["delta_margin_vs_full"] < 0.0)
        .groupby("coverage", as_index=False)
        .agg(harm_rate=("harm", "mean"), mean_delta_margin=("delta_margin_vs_full", "mean"), std_delta_margin=("delta_margin_vs_full", "std"))
        if not twotank.empty
        else pd.DataFrame(columns=["coverage", "harm_rate", "mean_delta_margin", "std_delta_margin"])
    )
    text = f"""# Signal-Set Ablation Seed Sweep

## Seeds

{", ".join(str(seed) for seed in summary["seeds"])}

## CSTR no-repair improvement

{_markdown_table(cstr_report, ["coverage", "win_rate", "mean_delta_margin", "std_delta_margin"])}

## TwoTank no-repair harm

{_markdown_table(twotank_report, ["coverage", "harm_rate", "mean_delta_margin", "std_delta_margin"])}

## Failed seeds

{summary["failed_seeds"] or "none"}

## Verdict

{summary["verdict"]}
"""
    _ensure_dir(output.parent)
    output.write_text(text, encoding="utf-8")


def make_repair_signal_role_decision_gate(
    config_path: str | Path,
    repair_validation_path: str | Path,
    repair_vs_invariant_path: str | Path,
    signal_ablation_path: str | Path,
    seed_sweep_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    config = load_repair_signal_semantics_config(config_path)
    repair_validation = _load_json(repair_validation_path)
    repair_vs_invariant = _load_json(repair_vs_invariant_path)
    signal_ablation = _load_json(signal_ablation_path)
    seed_sweep = _load_json(seed_sweep_path)
    decision, next_action = repair_role_decision(repair_validation, repair_vs_invariant, signal_ablation, seed_sweep)
    allowed_claim = "A weak but positive low-coverage result under the frozen protocol, with repair_amount treated according to the role decision gate."
    result = {
        "decision": decision,
        "allowed_next_action": next_action,
        "allowed_claim": allowed_claim,
        "expansion_allowed": False,
        "forbidden_next_actions": ["ADD_RSSM", "ADD_THIRD_SYSTEM", "ADD_NEW_JUDGE_FAMILY", "MUTATE_FROZEN_PROTOCOL"],
        "forbidden_claims": ["general reliability", "safety certification", "product readiness", "strong two-system support"],
        "inputs": {
            "repair_validation": repair_validation,
            "repair_vs_invariant": repair_vs_invariant,
            "signal_ablation": signal_ablation,
            "seed_sweep": seed_sweep,
        },
    }
    write_repair_role_decision_report(result, Path(output))
    _write_json(Path(output).with_suffix(".json"), result)
    if config["forbidden"]["allow_new_systems"] is not False:
        raise RuntimeError("decision gate config unexpectedly allows expansion")
    return result


def repair_role_decision(
    repair_validation: dict[str, Any],
    repair_vs_invariant: dict[str, Any],
    signal_ablation: dict[str, Any],
    seed_sweep: dict[str, Any],
) -> tuple[str, str]:
    rv = repair_validation.get("verdict")
    rvi = repair_vs_invariant.get("verdict")
    ablation = signal_ablation.get("verdict")
    seed = seed_sweep.get("verdict")
    cstr_delta = float(signal_ablation.get("cstr_no_repair_mean_delta_margin_vs_full", 0.0))
    cstr_no_hurt = cstr_delta >= -1e-12
    if rv == "REPAIR_IMPLEMENTATION_BUG":
        return "FIX_REPAIR_IMPLEMENTATION", "IMPLEMENT_REPAIR_BUG_FIX"
    if (
        rv == "REPAIR_CORRECT_BUT_CSTR_IRRELEVANT"
        and rvi in {"INVARIANT_DOMINATES_REPAIR", "REPAIR_SYSTEM_SPECIFIC"}
        and ablation == "NO_REPAIR_IMPROVES_CSTR_WITHOUT_HURTING_TWOTANK"
        and seed == "SEED_STABLE_NO_REPAIR_BENEFIT"
    ):
        return "MARK_REPAIR_SYSTEM_SPECIFIC", "IMPLEMENT_SYSTEM_SPECIFIC_SIGNAL_GATING"
    if rv == "REPAIR_CORRECT_BUT_CSTR_IRRELEVANT" and cstr_no_hurt:
        return "MARK_REPAIR_DIAGNOSTIC_ONLY_FOR_CSTR", "UPDATE_SIGNAL_SEMANTICS_ONLY"
    if rvi == "BOTH_USEFUL" and ablation == "NO_REPAIR_NO_BENEFIT":
        return "KEEP_REPAIR_UNIVERSAL", "KEEP_CURRENT_WEAK_CLAIM"
    return "INCONCLUSIVE", "DO_NOT_EXPAND"


def write_repair_role_decision_report(result: dict[str, Any], output: Path) -> None:
    inputs = result["inputs"]
    text = f"""# Repair Signal Role Decision Gate

## Starting point

CSTR weakness diagnosis identified REPAIR_SIGNAL_BLIND_SPOT.

## Controlled repair validation

{inputs["repair_validation"].get("verdict")}

## Repair vs invariant comparison

{inputs["repair_vs_invariant"].get("verdict")}

## Signal-set ablation

{inputs["signal_ablation"].get("verdict")}

## Seed robustness

{inputs["seed_sweep"].get("verdict")}

## Decision

{result["decision"]}

## Allowed next action

{result["allowed_next_action"]}

## Forbidden next actions

{", ".join(result["forbidden_next_actions"])}

## Allowed claim after this gate

{result["allowed_claim"]}

## Forbidden claims

{", ".join(result["forbidden_claims"])}
"""
    _ensure_dir(output.parent)
    output.write_text(text, encoding="utf-8")


def write_signal_semantics_report(output: str | Path) -> dict[str, str]:
    return write_signal_semantics_artifacts(output)
