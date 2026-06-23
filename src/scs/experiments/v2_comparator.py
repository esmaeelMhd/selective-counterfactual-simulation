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

from scs.experiments.v2 import (
    SIMPLE_RISK_COLUMN,
    _fit_stronger_baselines,
    _fit_v2_calibrated_scores,
    _load_json,
    _markdown_table,
    _prediction_rows_for_seed_system,
    _risk_curve_from_labels,
    _scan_forbidden_runtime_imports,
    _score_stronger_baseline,
    _target_rows,
    _valid_systems_for_v2,
    load_event_config,
    load_v2_config,
)


COMPARATOR_VERDICTS = {
    "CALIBRATED_BEATS_FAIR_DEPLOYABLE_BASELINE",
    "CALIBRATED_TARGET_DEPENDENT",
    "CALIBRATED_FAILS_FAIR_DEPLOYABLE_BASELINE",
    "COMPARATOR_ENVELOPE_TOO_STRICT_BUT_METHOD_WEAK",
    "INVALID_COMPARATOR_STATISTICS",
}
DECISION_LABELS = {
    "CALIBRATED_FAILS_FAIR_BASELINE",
    "CALIBRATED_BEATS_FAIR_BASELINE_ONLY",
    "CALIBRATED_TARGET_DEPENDENT",
    "COMPARATOR_TOO_STRICT_BUT_METHOD_STILL_WEAK",
    "INVALID_COMPARATOR_ANALYSIS",
}
FAIR_MODE = "per_system_target_calibration_selected_baseline"
ROW_WISE_MODE = "row_wise_strongest_baseline_envelope"
BEST_CALIBRATED_MODE = "best_calibrated_family_vs_per_system_target_baseline"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    _ensure_dir(path.parent)
    frame.to_csv(path, index=False)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_output(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:  # pragma: no cover - depends on local git state
        return f"unknown: {exc}"


def load_comparator_config(path: str | Path) -> dict[str, Any]:
    config = _load_yaml(path)
    required = {
        "audit_id",
        "source_branch",
        "source_commit",
        "source_artifacts",
        "v2_config",
        "v2_event_config",
        "primary_calibrated_judge",
        "baseline_judges",
        "calibrated_family_judges",
        "diagnostic_only",
        "primary_coverages",
        "badness_targets",
        "selection_modes",
        "statistical_rules",
        "forbidden",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"missing comparator config keys: {missing}")
    if config["audit_id"] != "v2_comparator_fairness":
        raise ValueError("unexpected comparator audit_id")
    forbidden = config["forbidden"]
    for key in [
        "new_systems",
        "new_models",
        "new_judges",
        "new_signals",
        "oracle_as_real_method",
        "test_labels_for_deployable_baselines",
        "overwrite_v2_source_artifacts",
    ]:
        if forbidden.get(key) is not False:
            raise ValueError(f"comparator config must set forbidden.{key}: false")
    if "oracle_error_rank" not in config["diagnostic_only"]:
        raise ValueError("oracle_error_rank must be diagnostic-only")
    if ROW_WISE_MODE not in config["diagnostic_only"]:
        raise ValueError("row-wise envelope must be diagnostic-only")
    return config


def _source_artifact_paths(config: dict[str, Any]) -> dict[str, Path]:
    return {str(name): Path(path) for name, path in config["source_artifacts"].items()}


def verify_comparator_preconditions(config_path: str | Path, output: str | Path) -> dict[str, Any]:
    config = load_comparator_config(config_path)
    out_dir = Path(output)
    _ensure_dir(out_dir)
    artifacts = _source_artifact_paths(config)
    missing = [str(path) for path in artifacts.values() if not path.exists()]
    hashes = {name: _sha256_file(path) for name, path in artifacts.items() if path.exists()}

    reasons: list[str] = []
    if missing:
        reasons.append("source artifacts missing")
    risk_columns_ok = False
    scenario_non_empty = False
    if artifacts["frozen_risk_coverage"].exists():
        risk = pd.read_csv(artifacts["frozen_risk_coverage"], nrows=5)
        required = {
            "system_id",
            "seed",
            "model_id",
            "badness_target",
            "bad_threshold",
            "coverage",
            "judge_id",
            "false_accept_rate",
            "baseline_judge",
            "baseline_far",
            "absolute_margin",
        }
        risk_columns_ok = required.issubset(risk.columns)
        if not risk_columns_ok:
            reasons.append("frozen risk CSV missing required columns")
    if artifacts["frozen_scenario_scores"].exists():
        scenario_non_empty = len(pd.read_csv(artifacts["frozen_scenario_scores"], nrows=2)) > 0
        if not scenario_non_empty:
            reasons.append("frozen scenario CSV empty")

    decision_text = artifacts["scientific_gate"].read_text(encoding="utf-8") if artifacts["scientific_gate"].exists() else ""
    v2_decision_ok = "NO_METHOD_CLAIM_BENCHMARK_ONLY" in decision_text
    if not v2_decision_ok:
        reasons.append("v2 gate decision is not NO_METHOD_CLAIM_BENCHMARK_ONLY")
    diagnosis_exists = artifacts["underperformance_diagnosis"].exists()
    if not diagnosis_exists:
        reasons.append("underperformance diagnosis missing")

    forbidden_scan = _scan_forbidden_runtime_imports()
    if forbidden_scan["old_repo_runtime_import_hits"] or forbidden_scan["path_hack_hits"]:
        reasons.append("forbidden runtime import or path hack detected")
    diagnostic_only_ok = (
        "oracle_error_rank" in config["diagnostic_only"]
        and ROW_WISE_MODE in config["diagnostic_only"]
        and "oracle_error_rank" not in config["baseline_judges"]
    )
    if not diagnostic_only_ok:
        reasons.append("oracle or row-wise envelope not diagnostic-only")

    source_artifact_modified = False
    status = _git_output(["status", "--short"])
    status_paths = {line[3:].strip() for line in status.splitlines() if len(line) >= 4}
    source_paths = {str(path) for path in artifacts.values()}
    source_artifact_modified = bool(status_paths.intersection(source_paths))
    if source_artifact_modified:
        reasons.append("v2 source artifacts are modified in working tree")

    verdict = "READY_FOR_COMPARATOR_FAIRNESS_AUDIT" if not reasons else "NOT_READY"
    result = {
        "audit_id": config["audit_id"],
        "source_branch": config["source_branch"],
        "source_commit": config["source_commit"],
        "working_tree_status": status,
        "source_artifacts": {name: str(path) for name, path in artifacts.items()},
        "missing_source_artifacts": missing,
        "risk_columns_ok": risk_columns_ok,
        "scenario_csv_non_empty": scenario_non_empty,
        "v2_decision_ok": v2_decision_ok,
        "underperformance_diagnosis_exists": diagnosis_exists,
        "diagnostic_only_ok": diagnostic_only_ok,
        "forbidden_dependency_scan": forbidden_scan,
        "source_artifact_modified": source_artifact_modified,
        "verdict": verdict,
        "reasons": reasons,
    }
    _write_json(out_dir / "source_artifact_hashes.json", hashes)
    _write_json(out_dir / "precondition_check.json", result)
    report = f"""# v2 Comparator Fairness Preconditions

## Verdict

{verdict}

## Source Artifacts

Checked: {len(artifacts)}
Missing: {missing or ["none"]}

## v2 Gate

Decision is `NO_METHOD_CLAIM_BENCHMARK_ONLY`: {v2_decision_ok}

## Diagnostic-Only Controls

Oracle and row-wise envelope diagnostic-only: {diagnostic_only_ok}

## Frozen Artifact Integrity

Frozen risk columns ok: {risk_columns_ok}
Frozen scenario CSV non-empty: {scenario_non_empty}
Source artifact modified in working tree: {source_artifact_modified}

## Dependency Scan

{forbidden_scan}

## Reasons

{reasons or ["none"]}
"""
    Path("reports/v2_comparator_fairness_preconditions.md").write_text(report, encoding="utf-8")
    return result


def _target_thresholds(v2_config: dict[str, Any], target: str) -> list[float | None]:
    if target == "bad_event":
        return [None]
    return [float(value) for value in v2_config["rmse_thresholds"]]


def _add_risk_columns_for_target(
    target_frame: pd.DataFrame,
    baseline_judges: list[str],
    calibrated_judges: list[str],
    primary_coverages: list[float],
    seed: int,
) -> pd.DataFrame:
    scored = target_frame.copy()
    calibration = scored[scored["role"] == "judge_calibration"].copy()
    if calibration.empty:
        raise RuntimeError("calibration rows are required for comparator selection")
    stronger = _fit_stronger_baselines(calibration, primary_coverages)
    for split_name, table in scored.groupby("split", sort=False):
        idx = table.index
        for judge_id in baseline_judges:
            if judge_id == "random_baseline":
                risk = np.random.default_rng(50000 + int(seed) + len(split_name)).uniform(0.0, 1.0, len(table))
            elif judge_id in SIMPLE_RISK_COLUMN:
                risk = table[SIMPLE_RISK_COLUMN[judge_id]].to_numpy(dtype=float)
            else:
                risk = _score_stronger_baseline(judge_id, stronger.get(judge_id), table)
            scored.loc[idx, f"risk_{judge_id}"] = risk
        calibrated_scores = _fit_v2_calibrated_scores(calibration, table, primary_coverages)
        for judge_id in calibrated_judges:
            scored.loc[idx, f"risk_{judge_id}"] = calibrated_scores[judge_id]
        scored.loc[idx, "risk_oracle_error_rank"] = table["badness_error"].to_numpy(dtype=float)
    return scored


def build_comparator_source_scores(config_path: str | Path, source_dir: str | Path) -> Path:
    """Reconstruct calibration+test risk scores in the comparator namespace.

    The frozen v2 scenario file stores judge-test rows only. Deployable
    comparator selection requires calibration rows, so this helper recomputes
    v2 scoring deterministically and writes only under the comparator output
    tree.
    """
    config = load_comparator_config(config_path)
    source_dir = Path(source_dir)
    source_path = source_dir / "comparator_source_scores.csv"
    if source_path.exists():
        return source_path
    v2_config = load_v2_config(config["v2_config"])
    event_config = load_event_config(config["v2_event_config"])
    frozen_output = Path(config["source_artifacts"]["frozen_risk_coverage"]).parent
    valid_systems = _valid_systems_for_v2(v2_config, frozen_output)
    frames: list[pd.DataFrame] = []
    primary_coverages = [float(value) for value in config["primary_coverages"]]
    for system_id in valid_systems:
        for seed in v2_config["seeds"]:
            base_scores, _, diagnostics = _prediction_rows_for_seed_system(system_id, int(seed), v2_config, event_config)
            if diagnostics["split_overlap"]["overlap_count"] > 0:
                raise RuntimeError(f"calibration/test overlap for {system_id} seed {seed}")
            for target in config["badness_targets"]:
                for threshold in _target_thresholds(v2_config, target):
                    target_frame = _target_rows(base_scores, target, threshold)
                    scored = _add_risk_columns_for_target(
                        target_frame,
                        baseline_judges=list(config["baseline_judges"]),
                        calibrated_judges=list(config["calibrated_family_judges"]),
                        primary_coverages=primary_coverages,
                        seed=int(seed),
                    )
                    keep_cols = [
                        "system_id",
                        "seed",
                        "role",
                        "split",
                        "scenario_type",
                        "scenario_id",
                        "model_id",
                        "rmse",
                        "true_event",
                        "pred_event",
                        "event_mismatch",
                        "badness_target",
                        "badness_error",
                        "bad_label",
                        "bad_threshold",
                        *[column for column in scored.columns if column.startswith("risk_")],
                    ]
                    frames.append(scored.assign(badness_target=target)[keep_cols])
    source = pd.concat(frames, ignore_index=True)
    if source.empty or source.isna().any().any():
        raise RuntimeError("reconstructed comparator source scores are empty or contain NaN")
    _write_csv(source_path, source)
    summary = {
        "verdict": "COMPARATOR_SOURCE_RECONSTRUCTED",
        "rows": int(len(source)),
        "systems": sorted(source["system_id"].unique().tolist()),
        "seeds": sorted(int(value) for value in source["seed"].unique().tolist()),
        "roles": sorted(source["role"].unique().tolist()),
        "uses_test_labels_for_deployable_selection": False,
    }
    _write_json(source_dir / "comparator_source_summary.json", summary)
    return source_path


def _selection_metric(table: pd.DataFrame, judge_id: str, coverage: float) -> dict[str, Any]:
    if table.empty:
        raise ValueError("selection table cannot be empty")
    risk_col = f"risk_{judge_id}"
    if risk_col not in table.columns:
        raise ValueError(f"missing risk column for {judge_id}")
    scores = table[risk_col].to_numpy(dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError(f"non-finite scores for {judge_id}")
    order = np.argsort(scores, kind="mergesort")
    n = len(table)
    accepted_count = min(max(int(math.ceil(float(coverage) * n)), 1), n)
    accepted = table.iloc[order[:accepted_count]]
    labels = accepted["bad_label"].astype(bool).to_numpy()
    errors = accepted["badness_error"].to_numpy(dtype=float)
    return {
        "selection_far": float(np.mean(labels)),
        "selection_accepted_error": float(np.mean(errors)),
        "accepted_count": int(accepted_count),
        "coverage_achieved": float(accepted_count / n),
    }


def select_judge_from_calibration(
    table: pd.DataFrame,
    candidate_judges: list[str],
    coverage: float,
) -> dict[str, Any]:
    if "role" in table.columns and set(table["role"].unique()) - {"judge_calibration"}:
        raise ValueError("deployable comparator selection must receive calibration rows only")
    rows = []
    for judge_id in candidate_judges:
        if judge_id == "oracle_error_rank":
            raise ValueError("oracle_error_rank cannot be selected as deployable")
        metric = _selection_metric(table, judge_id, coverage)
        rows.append({"judge_id": judge_id, **metric})
    ranked = sorted(
        rows,
        key=lambda row: (
            row["selection_far"],
            row["selection_accepted_error"],
            -row["coverage_achieved"],
            row["judge_id"],
        ),
    )
    best = ranked[0]
    tie_breaker_used = sum(
        row["selection_far"] == best["selection_far"]
        and row["selection_accepted_error"] == best["selection_accepted_error"]
        for row in ranked
    ) > 1
    return {
        "selected_judge_id": str(best["judge_id"]),
        "selection_metric": "false_accept_rate",
        "selection_far": float(best["selection_far"]),
        "selection_accepted_error": float(best["selection_accepted_error"]),
        "coverage_achieved": float(best["coverage_achieved"]),
        "tie_breaker_used": bool(tie_breaker_used),
    }


def _build_selection_rows(
    calibration: pd.DataFrame,
    candidate_judges: list[str],
    coverages: list[float],
    mode: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if mode == "global_calibration_selected_baseline":
        group_cols: list[str] = ["seed"]
        system_value = "ALL"
        target_value = "ALL"
    elif mode == "per_system_calibration_selected_baseline":
        group_cols = ["seed", "system_id"]
        system_value = None
        target_value = "ALL"
    elif mode in {"per_system_target_calibration_selected_baseline", "best_calibrated_family_selected_on_calibration"}:
        group_cols = ["seed", "system_id", "badness_target"]
        system_value = None
        target_value = None
    else:
        raise ValueError(f"unknown selection mode: {mode}")
    for keys, group in calibration.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = dict(zip(group_cols, keys, strict=True))
        for coverage in coverages:
            selected = select_judge_from_calibration(group, candidate_judges, float(coverage))
            rows.append(
                {
                    "selection_mode": mode,
                    "seed": int(key_map["seed"]),
                    "system_id": str(system_value if system_value is not None else key_map["system_id"]),
                    "badness_target": str(target_value if target_value is not None else key_map["badness_target"]),
                    "coverage": float(coverage),
                    **selected,
                    "source_split": "calibration",
                    "uses_test_labels": False,
                    "tie_breaker_used": bool(selected["tie_breaker_used"]),
                }
            )
    return pd.DataFrame(rows)


def build_comparator_selection_tables(config_path: str | Path, output: str | Path) -> dict[str, Any]:
    config = load_comparator_config(config_path)
    out_dir = Path(output)
    _ensure_dir(out_dir)
    source_path = build_comparator_source_scores(config_path, out_dir.parent / "rerun_source")
    source = pd.read_csv(source_path)
    calibration = source[source["role"] == "judge_calibration"].copy()
    if calibration.empty:
        raise RuntimeError("calibration rows missing; cannot select deployable comparators")
    if calibration["split"].str.contains("test", case=False).any():
        raise RuntimeError("test rows leaked into calibration selection")
    coverages = [float(value) for value in config["primary_coverages"]]
    baseline_judges = list(config["baseline_judges"])
    calibrated_family = list(config["calibrated_family_judges"])

    global_selection = _build_selection_rows(calibration, baseline_judges, coverages, "global_calibration_selected_baseline")
    per_system_selection = _build_selection_rows(calibration, baseline_judges, coverages, "per_system_calibration_selected_baseline")
    per_system_target_selection = _build_selection_rows(calibration, baseline_judges, coverages, "per_system_target_calibration_selected_baseline")
    calibrated_selection = _build_selection_rows(
        calibration,
        calibrated_family,
        coverages,
        "best_calibrated_family_selected_on_calibration",
    )
    outputs = {
        "global_baseline_selection.csv": global_selection,
        "per_system_baseline_selection.csv": per_system_selection,
        "per_system_target_baseline_selection.csv": per_system_target_selection,
        "calibrated_family_selection.csv": calibrated_selection,
    }
    reasons: list[str] = []
    for name, frame in outputs.items():
        if frame.empty:
            reasons.append(f"{name} empty")
        if frame["uses_test_labels"].astype(bool).any():
            reasons.append(f"{name} uses test labels")
        if (frame["selected_judge_id"] == "oracle_error_rank").any():
            reasons.append(f"{name} selected oracle")
        _write_csv(out_dir / name, frame)
    verdict = "COMPARATOR_SELECTION_VALID" if not reasons else "COMPARATOR_SELECTION_INVALID"
    summary = {
        "verdict": verdict,
        "source_scores": str(source_path),
        "selection_rows": {name: int(len(frame)) for name, frame in outputs.items()},
        "uses_test_labels": False,
        "oracle_selected": False,
        "reasons": reasons,
        "selected_judge_counts": {
            name: frame["selected_judge_id"].value_counts().to_dict() for name, frame in outputs.items()
        },
    }
    _write_json(out_dir / "comparator_selection_summary.json", summary)
    report = f"""# v2 Comparator Selection

## Verdict

{verdict}

## Selection Rule

All deployable baselines are selected on calibration rows only. The primary metric is lowest false accept rate at coverage 0.05 and 0.10. Tie-breakers are lower accepted error, higher achieved coverage, then alphabetical `judge_id`.

## Global Baseline Selection Preview

{_markdown_table(global_selection, ["selection_mode", "seed", "coverage", "selected_judge_id", "selection_far", "source_split", "uses_test_labels"], max_rows=12)}

## Per-System-Target Selection Preview

{_markdown_table(per_system_target_selection, ["selection_mode", "seed", "system_id", "badness_target", "coverage", "selected_judge_id", "selection_far"], max_rows=18)}

## Calibrated-Family Selection Preview

{_markdown_table(calibrated_selection, ["selection_mode", "seed", "system_id", "badness_target", "coverage", "selected_judge_id", "selection_far"], max_rows=18)}

## Reasons

{reasons or ["none"]}
"""
    Path("reports/v2_comparator_selection.md").write_text(report, encoding="utf-8")
    if verdict != "COMPARATOR_SELECTION_VALID":
        raise RuntimeError(verdict)
    return summary


def _selection_lookup(frame: pd.DataFrame, **keys: Any) -> str:
    mask = pd.Series(True, index=frame.index)
    for column, value in keys.items():
        mask &= frame[column] == value
    matches = frame[mask]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one selection for {keys}, found {len(matches)}")
    return str(matches.iloc[0]["selected_judge_id"])


def _risk_key(row: pd.Series, judge_id: str | None = None) -> tuple[Any, ...]:
    return (
        str(row["system_id"]),
        int(row["seed"]),
        str(row["model_id"]),
        str(row["badness_target"]),
        round(float(row["bad_threshold"]), 12),
        round(float(row["coverage"]), 12),
        str(judge_id if judge_id is not None else row["judge_id"]),
    )


def _risk_lookup(risk_index: dict[tuple[Any, ...], pd.Series], base_row: pd.Series, judge_id: str) -> pd.Series:
    key = _risk_key(base_row, judge_id)
    try:
        return risk_index[key]
    except KeyError as exc:
        raise RuntimeError(f"expected one risk row for {judge_id}, found 0") from exc


def _evaluation_row(
    base_row: pd.Series,
    baseline_row: pd.Series,
    comparator_mode: str,
    calibrated_judge_id: str,
    baseline_judge_id: str,
    uses_test_labels: bool,
    deployable: bool,
    diagnostic: bool,
) -> dict[str, Any]:
    calibrated_far = float(base_row["false_accept_rate"])
    baseline_far = float(baseline_row["false_accept_rate"])
    margin = baseline_far - calibrated_far
    return {
        "system_id": str(base_row["system_id"]),
        "seed": int(base_row["seed"]),
        "model_id": str(base_row["model_id"]),
        "badness_target": str(base_row["badness_target"]),
        "bad_threshold": float(base_row["bad_threshold"]),
        "coverage": float(base_row["coverage"]),
        "comparator_mode": comparator_mode,
        "calibrated_judge_id": calibrated_judge_id,
        "baseline_judge_id": baseline_judge_id,
        "calibrated_far": calibrated_far,
        "baseline_far": baseline_far,
        "absolute_margin": float(margin),
        "relative_margin": float(margin / baseline_far) if baseline_far > 0.0 else 0.0,
        "calibrated_accepted_count": int(base_row["accepted_count"]),
        "baseline_accepted_count": int(baseline_row["accepted_count"]),
        "calibrated_false_accept_count": int(base_row["false_accept_count"]),
        "baseline_false_accept_count": int(baseline_row["false_accept_count"]),
        "uses_test_labels_for_baseline_selection": bool(uses_test_labels),
        "is_deployable_comparator": bool(deployable),
        "is_diagnostic_comparator": bool(diagnostic),
    }


def evaluate_comparator_fairness(config_path: str | Path, selections: str | Path, output: str | Path) -> dict[str, Any]:
    config = load_comparator_config(config_path)
    out_dir = Path(output)
    _ensure_dir(out_dir)
    risk = pd.read_csv(config["source_artifacts"]["frozen_risk_coverage"])
    risk_index: dict[tuple[Any, ...], pd.Series] = {}
    for _, risk_row in risk.iterrows():
        key = _risk_key(risk_row)
        if key in risk_index:
            raise RuntimeError(f"duplicate frozen risk key: {key}")
        risk_index[key] = risk_row
    primary = str(config["primary_calibrated_judge"])
    coverages = [float(value) for value in config["primary_coverages"]]
    primary_rows = risk[(risk["judge_id"] == primary) & (risk["coverage"].isin(coverages))].copy()
    if primary_rows.empty:
        raise RuntimeError("primary calibrated rows missing from frozen risk coverage")
    selection_dir = Path(selections)
    global_sel = pd.read_csv(selection_dir / "global_baseline_selection.csv")
    system_sel = pd.read_csv(selection_dir / "per_system_baseline_selection.csv")
    system_target_sel = pd.read_csv(selection_dir / "per_system_target_baseline_selection.csv")
    calibrated_sel = pd.read_csv(selection_dir / "calibrated_family_selection.csv")
    rows: list[dict[str, Any]] = []
    for _, row in primary_rows.iterrows():
        baseline_judge = str(row["baseline_judge"])
        baseline_row = _risk_lookup(risk_index, row, baseline_judge)
        rows.append(
            _evaluation_row(
                row,
                baseline_row,
                ROW_WISE_MODE,
                primary,
                baseline_judge,
                uses_test_labels=True,
                deployable=False,
                diagnostic=True,
            )
        )
        selected_global = _selection_lookup(global_sel, seed=int(row["seed"]), coverage=float(row["coverage"]))
        rows.append(
            _evaluation_row(
                row,
                _risk_lookup(risk_index, row, selected_global),
                "global_calibration_selected_baseline",
                primary,
                selected_global,
                uses_test_labels=False,
                deployable=True,
                diagnostic=False,
            )
        )
        selected_system = _selection_lookup(
            system_sel,
            seed=int(row["seed"]),
            system_id=str(row["system_id"]),
            coverage=float(row["coverage"]),
        )
        rows.append(
            _evaluation_row(
                row,
                _risk_lookup(risk_index, row, selected_system),
                "per_system_calibration_selected_baseline",
                primary,
                selected_system,
                uses_test_labels=False,
                deployable=True,
                diagnostic=False,
            )
        )
        selected_target = _selection_lookup(
            system_target_sel,
            seed=int(row["seed"]),
            system_id=str(row["system_id"]),
            badness_target=str(row["badness_target"]),
            coverage=float(row["coverage"]),
        )
        rows.append(
            _evaluation_row(
                row,
                _risk_lookup(risk_index, row, selected_target),
                FAIR_MODE,
                primary,
                selected_target,
                uses_test_labels=False,
                deployable=True,
                diagnostic=False,
            )
        )
        selected_calibrated = _selection_lookup(
            calibrated_sel,
            seed=int(row["seed"]),
            system_id=str(row["system_id"]),
            badness_target=str(row["badness_target"]),
            coverage=float(row["coverage"]),
        )
        calibrated_row = _risk_lookup(risk_index, row, selected_calibrated)
        rows.append(
            _evaluation_row(
                calibrated_row,
                _risk_lookup(risk_index, row, selected_target),
                BEST_CALIBRATED_MODE,
                selected_calibrated,
                selected_target,
                uses_test_labels=False,
                deployable=True,
                diagnostic=False,
            )
        )
    by_row = pd.DataFrame(rows)
    if by_row.empty or by_row.isna().any().any():
        raise RuntimeError("comparator fairness evaluation is empty or contains NaN")
    summary = (
        by_row.groupby(["comparator_mode", "badness_target"], as_index=False)
        .agg(
            mean_calibrated_far=("calibrated_far", "mean"),
            mean_baseline_far=("baseline_far", "mean"),
            mean_absolute_margin=("absolute_margin", "mean"),
            mean_relative_margin=("relative_margin", "mean"),
            seed_win_rate=("absolute_margin", lambda values: float(np.mean(np.asarray(values) > 0.0))),
            row_count=("absolute_margin", "size"),
            deployable=("is_deployable_comparator", "first"),
            diagnostic=("is_diagnostic_comparator", "first"),
        )
        .sort_values(["comparator_mode", "badness_target"])
    )
    _write_csv(out_dir / "comparator_fairness_by_row.csv", by_row)
    _write_csv(out_dir / "comparator_fairness_summary.csv", summary)
    deployable_leakage = bool(by_row[by_row["is_deployable_comparator"]]["uses_test_labels_for_baseline_selection"].any())
    row_wise_diagnostic = bool(
        by_row[by_row["comparator_mode"] == ROW_WISE_MODE]["is_diagnostic_comparator"].all()
        and not by_row[by_row["comparator_mode"] == ROW_WISE_MODE]["is_deployable_comparator"].any()
    )
    verdict = "FAIR_BASELINE_ANALYSIS_COMPLETE" if not deployable_leakage and row_wise_diagnostic else "FAIR_BASELINE_ANALYSIS_INVALID"
    payload = {
        "verdict": verdict,
        "row_count": int(len(by_row)),
        "summary_rows": int(len(summary)),
        "deployable_baseline_selection_uses_test_labels": deployable_leakage,
        "row_wise_envelope_diagnostic_only": row_wise_diagnostic,
        "comparator_modes": sorted(by_row["comparator_mode"].unique().tolist()),
        "summary": summary.to_dict(orient="records"),
    }
    _write_json(out_dir / "comparator_fairness_summary.json", payload)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    plot = summary.pivot(index="comparator_mode", columns="badness_target", values="mean_absolute_margin").fillna(0.0)
    plot.plot(kind="bar", ax=ax)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_ylabel("Mean FAR margin")
    ax.set_xlabel("Comparator mode")
    ax.set_title("v2 comparator fairness margins")
    fig.tight_layout()
    fig.savefig(out_dir / "comparator_fairness_plot.png", dpi=160)
    plt.close(fig)
    report = f"""# v2 Comparator Fairness Evaluation

## Verdict

{verdict}

## Deployable Comparator Check

Deployable baseline selection uses test labels: {deployable_leakage}

## Diagnostic Envelope Check

Row-wise envelope diagnostic-only: {row_wise_diagnostic}

## RMSE and Event Summary

{_markdown_table(summary, ["comparator_mode", "badness_target", "mean_absolute_margin", "mean_calibrated_far", "mean_baseline_far", "deployable", "diagnostic"], max_rows=20)}
"""
    Path("reports/v2_comparator_fairness_evaluation.md").write_text(report, encoding="utf-8")
    if verdict != "FAIR_BASELINE_ANALYSIS_COMPLETE":
        raise RuntimeError(verdict)
    return payload


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_samples: int, confidence: float) -> tuple[float, float, np.ndarray]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return 0.0, 0.0, np.zeros(n_samples, dtype=float)
    samples = np.empty(n_samples, dtype=float)
    for idx in range(n_samples):
        samples[idx] = float(np.mean(rng.choice(values, size=len(values), replace=True)))
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(samples, alpha)), float(np.quantile(samples, 1.0 - alpha)), samples


def comparator_statistics_verdict(
    effect: pd.DataFrame,
    fair_mode: str = FAIR_MODE,
    min_margin: float = 0.05,
    min_seed_win_rate: float = 0.70,
) -> str:
    if effect.empty or fair_mode not in set(effect["comparator_mode"]):
        return "INVALID_COMPARATOR_STATISTICS"
    fair = effect[effect["comparator_mode"] == fair_mode].copy()
    systems = sorted(fair["system_id"].unique().tolist())
    if not systems:
        return "INVALID_COMPARATOR_STATISTICS"
    by_system = (
        fair.groupby("system_id", as_index=False)
        .agg(
            mean_far_margin=("mean_far_margin", "mean"),
            positive_ci=("positive_ci_excludes_zero", "any"),
            seed_win_rate=("seed_win_rate", "mean"),
            practical=("practical_threshold_pass", "any"),
        )
    )
    positive_systems = int((by_system["mean_far_margin"] > 0.0).sum())
    ci_positive_systems = int(by_system["positive_ci"].sum())
    robust_win_systems = int((by_system["seed_win_rate"] >= min_seed_win_rate).sum())
    event_worsening = bool(((fair["badness_target"] == "bad_event") & (fair["mean_far_margin"] < 0.0)).any())
    rmse_mean = float(fair[fair["badness_target"] == "bad_rmse"]["mean_far_margin"].mean())
    event_mean = float(fair[fair["badness_target"] == "bad_event"]["mean_far_margin"].mean())
    practical_count = int(by_system["practical"].sum())
    if (
        positive_systems >= 2
        and ci_positive_systems >= 2
        and robust_win_systems >= 2
        and practical_count >= 2
        and not event_worsening
    ):
        return "CALIBRATED_BEATS_FAIR_DEPLOYABLE_BASELINE"
    if rmse_mean > 0.0 and (event_mean <= 0.0 or event_worsening):
        return "CALIBRATED_TARGET_DEPENDENT"
    if positive_systems == 0 or float(by_system["mean_far_margin"].mean()) <= 0.0:
        return "CALIBRATED_FAILS_FAIR_DEPLOYABLE_BASELINE"
    if positive_systems > 0 and (
        positive_systems < 2
        or ci_positive_systems < 2
        or practical_count < 2
        or event_worsening
        or float(by_system["mean_far_margin"].mean()) < min_margin
    ):
        return "COMPARATOR_ENVELOPE_TOO_STRICT_BUT_METHOD_WEAK"
    return "CALIBRATED_FAILS_FAIR_DEPLOYABLE_BASELINE"


def run_comparator_statistical_audit(config_path: str | Path, evaluation: str | Path, output: str | Path) -> dict[str, Any]:
    config = load_comparator_config(config_path)
    out_dir = Path(output)
    _ensure_dir(out_dir)
    by_row = pd.read_csv(evaluation)
    if by_row.empty:
        raise RuntimeError("evaluation by-row CSV is empty")
    rules = config["statistical_rules"]
    confidence = float(rules["confidence_level"])
    n_boot = int(rules["bootstrap_iterations"])
    min_margin = float(rules["min_absolute_margin_for_meaningful_effect"])
    min_win = float(rules["min_seed_win_rate_for_robust_effect"])
    seed_margins = (
        by_row.groupby(["comparator_mode", "system_id", "seed", "badness_target"], as_index=False)
        .agg(
            mean_absolute_margin=("absolute_margin", "mean"),
            mean_relative_margin=("relative_margin", "mean"),
            deployable=("is_deployable_comparator", "first"),
            diagnostic=("is_diagnostic_comparator", "first"),
        )
        .sort_values(["comparator_mode", "system_id", "seed", "badness_target"])
    )
    rng = np.random.default_rng(8042)
    effect_rows: list[dict[str, Any]] = []
    boot_rows: list[dict[str, Any]] = []
    for (mode, system_id, target), group in seed_margins.groupby(["comparator_mode", "system_id", "badness_target"], sort=True):
        values = group["mean_absolute_margin"].to_numpy(dtype=float)
        ci_low, ci_high, samples = _bootstrap_ci(values, rng, n_boot, confidence)
        mean_margin = float(np.mean(values))
        seed_win_rate = float(np.mean(values > 0.0))
        practical = bool(mean_margin >= min_margin and seed_win_rate >= min_win and ci_low > 0.0)
        event_worsening = bool(target == "bad_event" and mean_margin < 0.0)
        effect_rows.append(
            {
                "comparator_mode": mode,
                "system_id": system_id,
                "badness_target": target,
                "mean_far_margin": mean_margin,
                "mean_relative_margin": float(group["mean_relative_margin"].mean()),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "seed_win_rate": seed_win_rate,
                "positive_ci_excludes_zero": bool(ci_low > 0.0),
                "ci_excludes_zero": bool(ci_low > 0.0 or ci_high < 0.0),
                "practical_threshold_pass": practical,
                "event_risk_worsening": event_worsening,
                "seed_count": int(group["seed"].nunique()),
            }
        )
        for sample_index, sample in enumerate(samples):
            boot_rows.append(
                {
                    "comparator_mode": mode,
                    "system_id": system_id,
                    "badness_target": target,
                    "sample_index": sample_index,
                    "mean_far_margin": float(sample),
                }
            )
    effect = pd.DataFrame(effect_rows)
    bootstrap = pd.DataFrame(boot_rows)
    _write_csv(out_dir / "comparator_seed_margins.csv", seed_margins)
    _write_csv(out_dir / "comparator_effect_size.csv", effect)
    _write_csv(out_dir / "comparator_bootstrap_samples.csv", bootstrap)
    verdict = comparator_statistics_verdict(effect, FAIR_MODE, min_margin, min_win)
    fair = effect[effect["comparator_mode"] == FAIR_MODE]
    row_wise = effect[effect["comparator_mode"] == ROW_WISE_MODE]
    rmse_result = {
        "mean_margin": float(fair[fair["badness_target"] == "bad_rmse"]["mean_far_margin"].mean()),
        "positive_system_count": int((fair[fair["badness_target"] == "bad_rmse"]["mean_far_margin"] > 0.0).sum()),
    }
    event_result = {
        "mean_margin": float(fair[fair["badness_target"] == "bad_event"]["mean_far_margin"].mean()),
        "event_risk_worsening_count": int(fair["event_risk_worsening"].sum()),
    }
    positive_systems = sorted(
        fair.groupby("system_id")["mean_far_margin"].mean().loc[lambda values: values > 0.0].index.tolist()
    )
    practical_pass_count = int(fair.groupby("system_id")["practical_threshold_pass"].any().sum())
    event_worsening_count = int(fair["event_risk_worsening"].sum())
    summary = {
        "verdict": verdict,
        "fair_mode": FAIR_MODE,
        "row_wise_mode": ROW_WISE_MODE,
        "positive_systems": positive_systems,
        "positive_system_count": len(positive_systems),
        "practical_threshold_pass_count": practical_pass_count,
        "event_risk_worsening_count": event_worsening_count,
        "rmse_target_result": rmse_result,
        "event_target_result": event_result,
        "row_wise_mean_margin": float(row_wise["mean_far_margin"].mean()) if not row_wise.empty else 0.0,
        "fair_mean_margin": float(fair["mean_far_margin"].mean()) if not fair.empty else 0.0,
        "confidence_level": confidence,
        "bootstrap_iterations": n_boot,
    }
    _write_json(out_dir / "comparator_statistical_summary.json", summary)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    plot = effect[effect["comparator_mode"].isin([FAIR_MODE, ROW_WISE_MODE, BEST_CALIBRATED_MODE])]
    pivot = plot.groupby(["comparator_mode", "badness_target"], as_index=False)["mean_far_margin"].mean()
    pivot.pivot(index="comparator_mode", columns="badness_target", values="mean_far_margin").fillna(0.0).plot(kind="bar", ax=ax)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_ylabel("Mean seed-level FAR margin")
    ax.set_xlabel("Comparator mode")
    ax.set_title("Comparator fairness statistical margins")
    fig.tight_layout()
    fig.savefig(out_dir / "comparator_effect_size_plot.png", dpi=160)
    plt.close(fig)
    report = f"""# v2 Comparator Fairness Statistical Audit

## Verdict

{verdict}

## Fair Deployable Baseline Effect

{_markdown_table(fair, ["system_id", "badness_target", "mean_far_margin", "bootstrap_ci_low", "bootstrap_ci_high", "seed_win_rate", "practical_threshold_pass"], max_rows=18)}

## Diagnostic Envelope Effect

{_markdown_table(row_wise, ["system_id", "badness_target", "mean_far_margin", "bootstrap_ci_low", "bootstrap_ci_high", "seed_win_rate"], max_rows=18)}

## RMSE Target Result

{rmse_result}

## Event-Risk Target Result

{event_result}
"""
    Path("reports/v2_comparator_fairness_statistical_audit.md").write_text(report, encoding="utf-8")
    if verdict == "INVALID_COMPARATOR_STATISTICS":
        raise RuntimeError(verdict)
    return summary


def decision_from_statistics(statistics: dict[str, Any]) -> tuple[str, str]:
    stat_verdict = str(statistics.get("verdict"))
    event_worsening = int(statistics.get("event_risk_worsening_count", 0)) > 0
    if stat_verdict == "CALIBRATED_BEATS_FAIR_DEPLOYABLE_BASELINE" and not event_worsening:
        return (
            "CALIBRATED_BEATS_FAIR_BASELINE_ONLY",
            "Calibrated refusal improves only against fair deployable baselines, not diagnostic envelope.",
        )
    if stat_verdict == "CALIBRATED_TARGET_DEPENDENT" or event_worsening:
        return (
            "CALIBRATED_TARGET_DEPENDENT",
            "Calibrated refusal is target-dependent and not reliable for event-risk.",
        )
    if stat_verdict == "COMPARATOR_ENVELOPE_TOO_STRICT_BUT_METHOD_WEAK":
        return (
            "COMPARATOR_TOO_STRICT_BUT_METHOD_STILL_WEAK",
            "Comparator envelope was too strict, but method evidence remains weak.",
        )
    if stat_verdict == "CALIBRATED_FAILS_FAIR_DEPLOYABLE_BASELINE":
        return (
            "CALIBRATED_FAILS_FAIR_BASELINE",
            "Benchmark exposes calibrated underperformance against fair baselines.",
        )
    return (
        "INVALID_COMPARATOR_ANALYSIS",
        "Comparator analysis invalid; no claim update allowed.",
    )


def make_comparator_decision_gate(
    config_path: str | Path,
    selection: str | Path,
    evaluation: str | Path,
    statistics: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    config = load_comparator_config(config_path)
    selection_summary = _load_json(selection)
    evaluation_summary = _load_json(evaluation)
    stat_summary = _load_json(statistics)
    decision, allowed_claim = decision_from_statistics(stat_summary)
    if decision not in DECISION_LABELS:
        raise RuntimeError(f"invalid comparator decision: {decision}")
    starting_decision = "NO_METHOD_CLAIM_BENCHMARK_ONLY"
    no_upgrade = decision in {
        "CALIBRATED_FAILS_FAIR_BASELINE",
        "CALIBRATED_TARGET_DEPENDENT",
        "COMPARATOR_TOO_STRICT_BUT_METHOD_STILL_WEAK",
        "INVALID_COMPARATOR_ANALYSIS",
    }
    forbidden_claims = [
        "row-wise strongest-baseline envelope is deployable",
        "calibrated refusal works generally",
        "event-risk refusal is reliable",
        "safety certification",
        "trusted simulator",
        "product-ready digital twin",
    ]
    report = f"""# v2 Comparator Fairness Decision Gate

## Starting v2 decision

{starting_decision}

## Comparator taxonomy

Row-wise strongest-baseline envelope is diagnostic only. Deployable baselines are selected from calibration rows only.

## Selection validity

Selection verdict: {selection_summary.get("verdict")}

Deployable selection uses test labels: {selection_summary.get("uses_test_labels")}

## Fair comparator results

Fair mode: {stat_summary.get("fair_mode")}

Fair mean margin: {stat_summary.get("fair_mean_margin")}

Positive systems: {stat_summary.get("positive_systems")}

## Diagnostic envelope results

Row-wise mode: {stat_summary.get("row_wise_mode")}

Row-wise mean margin: {stat_summary.get("row_wise_mean_margin")}

## RMSE vs event-risk

RMSE target result: {stat_summary.get("rmse_target_result")}

Event target result: {stat_summary.get("event_target_result")}

## Decision

{decision}

## Allowed claim

{allowed_claim}

## Forbidden claims

{chr(10).join(f"- {claim}" for claim in forbidden_claims)}

## Recommended next action

Do not expand systems or claims until the event-risk failure mode and fair-baseline result are understood.
"""
    output_path = Path(output)
    output_path.write_text(report, encoding="utf-8")
    payload = {
        "decision": decision,
        "allowed_claim": allowed_claim,
        "scientific_claim_upgraded": not no_upgrade and decision == "CALIBRATED_BEATS_FAIR_BASELINE_ONLY",
        "starting_v2_decision": starting_decision,
        "selection_verdict": selection_summary.get("verdict"),
        "evaluation_verdict": evaluation_summary.get("verdict"),
        "statistical_verdict": stat_summary.get("verdict"),
        "row_wise_envelope_deployable": False,
        "deployable_baseline_selection_uses_test_labels": False,
    }
    _write_json(output_path.with_suffix(".json"), payload)
    write_comparator_docs(config, payload, stat_summary, selection_summary, evaluation_summary)
    return payload


def write_comparator_docs(
    config: dict[str, Any],
    decision: dict[str, Any],
    statistics: dict[str, Any],
    selection: dict[str, Any],
    evaluation: dict[str, Any],
) -> None:
    _ensure_dir(Path("docs/v2"))
    summary = f"""# v2 Comparator Fairness Summary

## Decision

{decision["decision"]}

## Allowed Claim

{decision["allowed_claim"]}

## Comparator Scope

This audit compares the primary calibrated judge to deployable fixed baselines selected on calibration rows and to the diagnostic row-wise strongest-baseline envelope.

## Main Results

- fair deployable baseline mean margin: {statistics.get("fair_mean_margin")}
- diagnostic envelope mean margin: {statistics.get("row_wise_mean_margin")}
- event-risk worsening count: {statistics.get("event_risk_worsening_count")}
- RMSE target result: {statistics.get("rmse_target_result")}
- event target result: {statistics.get("event_target_result")}

## Interpretation

The row-wise envelope remains diagnostic only. No broad method claim is allowed unless the decision gate says so explicitly.
"""
    Path("docs/v2/v2_comparator_fairness_summary.md").write_text(summary, encoding="utf-8")
    claim_rows = [
        [
            "calibrated method fails against row-wise envelope",
            "diagnostic-only evidence",
            "comparator fairness evaluation",
            "The calibrated candidate is compared against a diagnostic upper-bound envelope.",
        ],
        [
            "row-wise envelope is deployable",
            "forbidden",
            "comparator taxonomy",
            "The row-wise envelope is diagnostic only, not deployable.",
        ],
        [
            "calibrated method fails against fair fixed baseline",
            "decision-gated",
            "comparator decision gate",
            decision["allowed_claim"],
        ],
        [
            "calibrated method helps RMSE but not events",
            "target-dependent if supported by statistics",
            "RMSE and event target results",
            "Only state the target-specific result reported by the audit.",
        ],
        [
            "conformal baseline dominates",
            "baseline-specific observation only",
            "selection tables and frozen risk coverage",
            "A named baseline may dominate within this benchmark; do not generalize.",
        ],
        [
            "event-risk is main failure mode",
            "allowed if event rows show worsening",
            "event-risk target result",
            "Event-risk remains a separately reported failure mode.",
        ],
        [
            "benchmark exposes method failure",
            "allowed when decision says failure or target dependence",
            "decision gate",
            decision["allowed_claim"],
        ],
    ]
    lines = ["# v2 Comparator Claim Audit", "", "| Claim | Status | Evidence | Allowed wording |", "|---|---|---|---|"]
    lines.extend("| " + " | ".join(row) + " |" for row in claim_rows)
    Path("docs/v2/v2_comparator_claim_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    release = f"""# Release Note: v2 Comparator Fairness

Decision: {decision["decision"]}

Allowed claim: {decision["allowed_claim"]}

Selection verdict: {selection.get("verdict")}

Evaluation verdict: {evaluation.get("verdict")}

Statistical verdict: {statistics.get("verdict")}

Row-wise strongest-baseline envelope is diagnostic only. Deployable baselines are selected using calibration rows only.
"""
    Path("reports/release_note_v2_comparator_fairness.md").write_text(release, encoding="utf-8")


def report_contains_literal_numbers(report_path: Path, csv_path: Path, column: str) -> bool:
    report = report_path.read_text(encoding="utf-8")
    frame = pd.read_csv(csv_path)
    if frame.empty:
        return False
    value = frame[column].iloc[0]
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}" in report or str(float(value)) in report
    return str(value) in report


def scan_no_claim_inflation(paths: list[Path], decision: str) -> list[str]:
    forbidden = [
        r"\bworks generally\b",
        r"\btrustworthy\b",
        r"\bsafety certification\b",
        r"\btrusted simulator\b",
        r"\bdeployable row-wise\b",
        r"\brow-wise envelope is deployable\b",
    ]
    hits: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8").lower()
        if "row-wise envelope is deployable" in text and decision != "CALIBRATED_BEATS_FAIR_BASELINE_ONLY":
            hits.append(str(path))
        for pattern in forbidden:
            if re.search(pattern, text):
                hits.append(str(path))
                break
    return sorted(set(hits))
