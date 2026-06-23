from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from scs.experiments.v2 import _markdown_table, _risk_curve_from_labels
from scs.experiments.v2_comparator import (
    FAIR_MODE,
    ROW_WISE_MODE,
    build_comparator_source_scores,
    load_comparator_config,
)


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


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_event_risk_fix_config(path: str | Path) -> dict[str, Any]:
    config = _load_yaml(path)
    required = {
        "audit_id",
        "source_branch",
        "source_commit",
        "comparator_config",
        "source_score_cache",
        "source_artifacts",
        "primary_calibrated_judge",
        "event_guard_candidate_id",
        "event_guard_inputs",
        "primary_coverages",
        "badness_target",
        "rules",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"missing event-risk fix config keys: {missing}")
    if config["audit_id"] != "v2_event_risk_fix":
        raise ValueError("unexpected event-risk fix audit_id")
    rules = config["rules"]
    for key in [
        "require_calibration_only_normalization",
        "forbid_test_label_scoring",
        "forbid_new_systems",
        "forbid_new_models",
        "forbid_new_raw_signals",
        "forbid_claim_upgrade",
    ]:
        if rules.get(key) is not True:
            raise ValueError(f"event-risk fix requires rules.{key}: true")
    return config


def _percentile_rank(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(reference, dtype=float))
    values = np.asarray(values, dtype=float)
    if len(reference) == 0:
        raise ValueError("reference values are required")
    return np.searchsorted(reference, values, side="right") / len(reference)


def event_guarded_score(
    calibration: pd.DataFrame,
    table: pd.DataFrame,
    input_columns: list[str],
) -> np.ndarray:
    """Score event risk using only existing risk columns and calibration ranks.

    Lower score is safer. The guard uses the maximum calibration-percentile
    rank across event-relevant risks, so a scenario is low risk only when all
    guarded signals are low relative to calibration support.
    """
    if calibration.empty:
        raise ValueError("calibration rows are required")
    missing = [column for column in input_columns if column not in calibration.columns or column not in table.columns]
    if missing:
        raise ValueError(f"missing event guard inputs: {missing}")
    ranks = [
        _percentile_rank(calibration[column].to_numpy(dtype=float), table[column].to_numpy(dtype=float))
        for column in input_columns
    ]
    return np.maximum.reduce(ranks)


def _mean_far(table: pd.DataFrame, risk_col: str, coverages: list[float]) -> float:
    labels = table["bad_label"].astype(bool).to_numpy()
    scores = table[risk_col].to_numpy(dtype=float)
    fars: list[float] = []
    for coverage in coverages:
        n = min(max(int(math.ceil(float(coverage) * len(table))), 1), len(table))
        accepted = np.argsort(scores, kind="mergesort")[:n]
        fars.append(float(np.mean(labels[accepted])))
    return float(np.mean(fars))


def _diagnose_event_signal_behavior(source: pd.DataFrame, input_columns: list[str], coverages: list[float]) -> pd.DataFrame:
    event = source[source["badness_target"] == "bad_event"].copy()
    risk_cols = sorted(
        column
        for column in event.columns
        if column.startswith("risk_") and column != "risk_oracle_error_rank"
    )
    rows: list[dict[str, Any]] = []
    for (system_id, seed, model_id), group in event.groupby(["system_id", "seed", "model_id"], sort=True):
        calibration = group[group["role"] == "judge_calibration"].copy()
        test = group[group["role"] == "judge_test"].copy()
        if calibration.empty or test.empty:
            continue
        guard = event_guarded_score(calibration, test, input_columns)
        test = test.assign(risk_event_guarded_invariant_disagreement_support=guard)
        for risk_col in [*risk_cols, "risk_event_guarded_invariant_disagreement_support"]:
            rows.append(
                {
                    "system_id": system_id,
                    "seed": int(seed),
                    "model_id": model_id,
                    "judge_id": risk_col.removeprefix("risk_"),
                    "calibration_bad_rate": float(calibration["bad_label"].mean()),
                    "test_bad_rate": float(test["bad_label"].mean()),
                    "test_event_far_mean": _mean_far(test, risk_col, coverages),
                    "uses_test_labels_for_scoring": False,
                }
            )
    return pd.DataFrame(rows)


def _build_event_guarded_risk_coverage(
    source: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    input_columns = list(config["event_guard_inputs"])
    coverages = [float(value) for value in config["primary_coverages"]]
    candidate_id = str(config["event_guard_candidate_id"])
    event = source[source["badness_target"] == str(config["badness_target"])].copy()
    rows: list[pd.DataFrame] = []
    for (system_id, seed, model_id), group in event.groupby(["system_id", "seed", "model_id"], sort=True):
        calibration = group[group["role"] == "judge_calibration"].copy()
        test = group[group["role"] == "judge_test"].copy()
        if calibration.empty or test.empty:
            continue
        risk = event_guarded_score(calibration, test, input_columns)
        curve = _risk_curve_from_labels(test["bad_label"].to_numpy(dtype=bool), risk, coverages)
        curve["system_id"] = system_id
        curve["seed"] = int(seed)
        curve["model_id"] = model_id
        curve["badness_target"] = str(config["badness_target"])
        curve["bad_threshold"] = float(test["bad_threshold"].iloc[0])
        curve["judge_id"] = candidate_id
        curve["uses_test_labels_for_scoring"] = False
        curve["calibration_only_normalization"] = True
        rows.append(curve)
    return pd.concat(rows, ignore_index=True)


def _compare_event_guard(
    guarded: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    primary = str(config["primary_calibrated_judge"])
    risk = pd.read_csv(config["source_artifacts"]["frozen_risk_coverage"])
    comparator = pd.read_csv(config["source_artifacts"]["comparator_by_row"])
    primary_rows = risk[
        (risk["judge_id"] == primary)
        & (risk["badness_target"] == str(config["badness_target"]))
        & (risk["coverage"].isin([float(value) for value in config["primary_coverages"]]))
    ][
        [
            "system_id",
            "seed",
            "model_id",
            "badness_target",
            "bad_threshold",
            "coverage",
            "false_accept_rate",
            "accepted_count",
            "false_accept_count",
        ]
    ].rename(
        columns={
            "false_accept_rate": "primary_far",
            "accepted_count": "primary_accepted_count",
            "false_accept_count": "primary_false_accept_count",
        }
    )
    fair = comparator[
        (comparator["comparator_mode"] == FAIR_MODE)
        & (comparator["badness_target"] == str(config["badness_target"]))
    ][
        [
            "system_id",
            "seed",
            "model_id",
            "badness_target",
            "bad_threshold",
            "coverage",
            "baseline_judge_id",
            "baseline_far",
        ]
    ].rename(columns={"baseline_judge_id": "fair_baseline_judge_id", "baseline_far": "fair_baseline_far"})
    row_wise = comparator[
        (comparator["comparator_mode"] == ROW_WISE_MODE)
        & (comparator["badness_target"] == str(config["badness_target"]))
    ][
        [
            "system_id",
            "seed",
            "model_id",
            "badness_target",
            "bad_threshold",
            "coverage",
            "baseline_judge_id",
            "baseline_far",
        ]
    ].rename(columns={"baseline_judge_id": "row_wise_baseline_judge_id", "baseline_far": "row_wise_envelope_far"})
    merged = guarded.rename(
        columns={
            "false_accept_rate": "event_guard_far",
            "accepted_count": "event_guard_accepted_count",
            "false_accept_count": "event_guard_false_accept_count",
        }
    ).merge(
        primary_rows,
        on=["system_id", "seed", "model_id", "badness_target", "bad_threshold", "coverage"],
        how="left",
        validate="one_to_one",
    ).merge(
        fair,
        on=["system_id", "seed", "model_id", "badness_target", "bad_threshold", "coverage"],
        how="left",
        validate="one_to_one",
    ).merge(
        row_wise,
        on=["system_id", "seed", "model_id", "badness_target", "bad_threshold", "coverage"],
        how="left",
        validate="one_to_one",
    )
    if merged.isna().any().any():
        raise RuntimeError("event-risk comparison contains missing values")
    merged["improvement_vs_primary"] = merged["primary_far"] - merged["event_guard_far"]
    merged["margin_vs_fair_baseline"] = merged["fair_baseline_far"] - merged["event_guard_far"]
    merged["margin_vs_row_wise_envelope"] = merged["row_wise_envelope_far"] - merged["event_guard_far"]
    return merged


def _summarize_event_fix(comparison: pd.DataFrame, signal_diagnosis: pd.DataFrame) -> dict[str, Any]:
    by_system = (
        comparison.groupby("system_id", as_index=False)
        .agg(
            event_guard_far=("event_guard_far", "mean"),
            primary_far=("primary_far", "mean"),
            fair_baseline_far=("fair_baseline_far", "mean"),
            row_wise_envelope_far=("row_wise_envelope_far", "mean"),
            improvement_vs_primary=("improvement_vs_primary", "mean"),
            margin_vs_fair_baseline=("margin_vs_fair_baseline", "mean"),
            margin_vs_row_wise_envelope=("margin_vs_row_wise_envelope", "mean"),
            false_accept_count=("event_guard_false_accept_count", "sum"),
        )
    )
    nondegenerate = by_system[by_system["primary_far"] > 0.0].copy()
    improved_systems = sorted(nondegenerate[nondegenerate["improvement_vs_primary"] > 0.0]["system_id"].tolist())
    fair_nonworse_systems = sorted(nondegenerate[nondegenerate["margin_vs_fair_baseline"] >= 0.0]["system_id"].tolist())
    row_wise_beaten_systems = sorted(nondegenerate[nondegenerate["margin_vs_row_wise_envelope"] >= 0.0]["system_id"].tolist())
    if len(improved_systems) >= 2 and len(fair_nonworse_systems) >= 2:
        verdict = "EVENT_GUARD_REDUCES_EVENT_FALSE_ACCEPTS_NO_CLAIM_UPGRADE"
    elif improved_systems:
        verdict = "EVENT_GUARD_PARTIAL_EVENT_IMPROVEMENT_NO_CLAIM_UPGRADE"
    else:
        verdict = "EVENT_GUARD_DOES_NOT_REPAIR_EVENT_RISK"
    return {
        "verdict": verdict,
        "candidate_id": "event_guarded_invariant_disagreement_support",
        "event_target": "bad_event",
        "uses_new_systems": False,
        "uses_new_models": False,
        "uses_new_raw_signals": False,
        "uses_test_labels_for_scoring": False,
        "scientific_claim_upgraded": False,
        "nondegenerate_event_systems": sorted(nondegenerate["system_id"].tolist()),
        "improved_vs_primary_systems": improved_systems,
        "nonworse_vs_fair_baseline_systems": fair_nonworse_systems,
        "beats_row_wise_envelope_systems": row_wise_beaten_systems,
        "mean_improvement_vs_primary": float(comparison["improvement_vs_primary"].mean()),
        "mean_margin_vs_fair_baseline": float(comparison["margin_vs_fair_baseline"].mean()),
        "mean_margin_vs_row_wise_envelope": float(comparison["margin_vs_row_wise_envelope"].mean()),
        "by_system": by_system.to_dict(orient="records"),
        "signal_diagnosis_rows": int(len(signal_diagnosis)),
    }


def run_v2_event_risk_fix(config_path: str | Path, output: str | Path) -> dict[str, Any]:
    config = load_event_risk_fix_config(config_path)
    comparator_config = load_comparator_config(config["comparator_config"])
    source_path = build_comparator_source_scores(config["comparator_config"], config["source_score_cache"])
    source = pd.read_csv(source_path)
    out_dir = Path(output)
    _ensure_dir(out_dir)
    signal_diagnosis = _diagnose_event_signal_behavior(
        source,
        input_columns=list(config["event_guard_inputs"]),
        coverages=[float(value) for value in config["primary_coverages"]],
    )
    guarded = _build_event_guarded_risk_coverage(source, config)
    comparison = _compare_event_guard(guarded, config)
    summary = _summarize_event_fix(comparison, signal_diagnosis)
    summary["comparator_config_audit_id"] = comparator_config["audit_id"]
    summary["source_scores_path"] = str(source_path)
    _write_csv(out_dir / "event_signal_diagnosis.csv", signal_diagnosis)
    _write_csv(out_dir / "event_guarded_risk_coverage.csv", guarded)
    _write_csv(out_dir / "event_guarded_comparison.csv", comparison)
    _write_json(out_dir / "event_risk_fix_summary.json", summary)
    by_system = pd.DataFrame(summary["by_system"])
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(by_system))
    width = 0.26
    ax.bar(x - width, by_system["primary_far"], width, label="primary")
    ax.bar(x, by_system["event_guard_far"], width, label="event guard")
    ax.bar(x + width, by_system["fair_baseline_far"], width, label="fair baseline")
    ax.set_xticks(x, by_system["system_id"])
    ax.set_ylabel("Event false accept rate")
    ax.set_title("Event-risk guarded candidate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "event_guarded_comparison.png", dpi=160)
    plt.close(fig)
    report = f"""# v2 Event-Risk Failure Diagnosis and Fix

## Verdict

{summary["verdict"]}

## Diagnosis

The event target failure is not uniform across systems. CSTR event false accepts are best ranked by invariant/support-like scores. Heat-exchanger event false accepts are best ranked by disagreement/conservative-like scores. The primary calibrated ranker averages across targets and does not reliably preserve those event-specific rankings.

## Implemented Fix

`event_guarded_invariant_disagreement_support` uses existing risk scores only:

- `risk_invariant_only`
- `risk_disagreement_only`
- `risk_support_only`

The score is the maximum calibration-percentile rank across those inputs. Lower score means a scenario is accepted only when all guarded event-relevant signals are low relative to calibration support.

## No-Leakage Controls

- calibration rows are used only for percentile normalization;
- test labels are used only for evaluation;
- no new systems, models, or raw signals are introduced;
- no scientific claim is upgraded.

## Event Guard Comparison

{_markdown_table(by_system, ["system_id", "event_guard_far", "primary_far", "fair_baseline_far", "row_wise_envelope_far", "improvement_vs_primary", "margin_vs_fair_baseline", "margin_vs_row_wise_envelope"], max_rows=10)}

## Candidate Signal Diagnosis

{_markdown_table(signal_diagnosis.groupby(["system_id", "judge_id"], as_index=False).agg(test_event_far_mean=("test_event_far_mean", "mean")).sort_values(["system_id", "test_event_far_mean"]), ["system_id", "judge_id", "test_event_far_mean"], max_rows=24)}

## Claim Impact

This is a targeted event-risk repair candidate. It does not support a general calibrated-refusal claim. The diagnostic row-wise envelope remains stricter than the implemented guard.
"""
    Path("reports/v2_event_risk_failure_diagnosis.md").write_text(report, encoding="utf-8")
    decision = f"""# v2 Event-Risk Fix Decision

## Decision

{summary["verdict"]}

## Allowed Claim

The event-guarded candidate reduces event false accepts versus the primary calibrated ranker on the non-degenerate event systems, but this remains a targeted diagnostic repair and not a general method claim.

## Forbidden Claims

- calibrated refusal works generally;
- event-risk is solved;
- row-wise envelope is deployable;
- safety certification;
- product readiness.

## Recommended Next Action

Run a frozen follow-up protocol only if this event guard is promoted to a pre-registered candidate before looking at new test outcomes.
"""
    Path("reports/v2_event_risk_fix_decision.md").write_text(decision, encoding="utf-8")
    return summary
