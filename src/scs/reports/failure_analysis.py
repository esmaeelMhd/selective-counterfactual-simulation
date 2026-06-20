from __future__ import annotations

import json
import math
import re
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SIMPLE_JUDGES = [
    "support_only",
    "uncertainty_only",
    "disagreement_only",
    "invariant_only",
    "repair_only",
    "random_baseline",
]
REAL_JUDGES = [*SIMPLE_JUDGES, "combined_linear"]
PLOT_JUDGES = [*REAL_JUDGES, "oracle_error_rank"]
SIGNALS = [
    "support_distance",
    "uncertainty_score",
    "disagreement_score",
    "invariant_residual",
    "repair_amount",
    "combined_linear_score",
]
BASE_SIGNALS = [
    "support_distance",
    "uncertainty_score",
    "disagreement_score",
    "invariant_residual",
    "repair_amount",
]
TARGETS = [
    "rmse",
    "mae",
    "max_abs_error",
    "final_state_error",
    "event_error",
    "bad_rmse_label",
    "bad_event_label",
]
REQUIRED_RESULT_FILES = [
    "risk_coverage.csv",
    "scenario_scores.csv",
    "model_metrics.csv",
    "summary.json",
]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


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


def _decision_from_gate_text(text: str) -> str:
    match = re.search(r"## Decision\s+([A-Z_]+)", text)
    if match:
        return match.group(1)
    for decision in ["PROCEED_TO_CSTR", "FIX_V0_FIRST", "KILL_OR_DOWNGRADE_CLAIM"]:
        if decision in text:
            return decision
    raise ValueError("could not find decision in gate report")


def _required_numeric_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if pd.api.types.is_numeric_dtype(df[column])]


def verify_failed_gate(results: str | Path, gate: str | Path) -> dict[str, Any]:
    results_dir = Path(results)
    gate_path = Path(gate)
    out_dir = Path("results/failure_analysis")
    _ensure_dir(out_dir)
    report_path = Path("reports/failure_analysis_gate_verification.md")
    _ensure_dir(report_path.parent)

    artifacts = {}
    ready = True
    for name in REQUIRED_RESULT_FILES:
        path = results_dir / name
        artifacts[name] = {"exists": path.exists(), "size": path.stat().st_size if path.exists() else 0}
        ready = ready and artifacts[name]["exists"] and artifacts[name]["size"] > 0
    if not gate_path.exists():
        decision = "MISSING"
        ready = False
    else:
        decision = _decision_from_gate_text(gate_path.read_text(encoding="utf-8"))

    integrity: dict[str, Any] = {}
    for name in ["risk_coverage.csv", "scenario_scores.csv", "model_metrics.csv"]:
        path = results_dir / name
        if not path.exists():
            integrity[name] = {"non_empty": False, "numeric_nan_count": None}
            continue
        frame = pd.read_csv(path)
        numeric = frame.select_dtypes(include=[float, int])
        nan_count = int(numeric.isna().sum().sum())
        integrity[name] = {"non_empty": not frame.empty, "numeric_nan_count": nan_count}
        ready = ready and (not frame.empty) and nan_count == 0

    expansion_status = "ALLOWED" if decision == "PROCEED_TO_CSTR" else "BLOCKED"
    verdict = "READY_FOR_FAILURE_ANALYSIS" if ready and expansion_status == "BLOCKED" else "NOT_READY"
    result = {
        "results_dir": str(results_dir),
        "gate": str(gate_path),
        "decision": decision,
        "required_artifacts": artifacts,
        "data_integrity": integrity,
        "expansion_status": expansion_status,
        "verdict": verdict,
    }
    _write_json(out_dir / "gate_verification.json", result)
    artifact_df = pd.DataFrame([{"artifact": key, **value} for key, value in artifacts.items()])
    integrity_df = pd.DataFrame([{"file": key, **value} for key, value in integrity.items()])
    report = f"""# Failure Analysis Gate Verification

## Decision gate status

{decision}

## Required artifacts

{_markdown_table(artifact_df, ["artifact", "exists", "size"])}

## Data integrity checks

{_markdown_table(integrity_df, ["file", "non_empty", "numeric_nan_count"])}

## Expansion status

{expansion_status}

## Verdict

{verdict}
"""
    report_path.write_text(report, encoding="utf-8")
    if verdict != "READY_FOR_FAILURE_ANALYSIS":
        raise RuntimeError(f"failed gate verification verdict={verdict}")
    return result


def _threshold_from_results(results_dir: Path) -> float:
    risk = pd.read_csv(results_dir / "risk_coverage.csv")
    if "threshold" not in risk:
        raise ValueError("risk_coverage.csv missing threshold column")
    thresholds = sorted(risk["threshold"].dropna().unique())
    if len(thresholds) != 1:
        raise ValueError(f"expected exactly one threshold, found {thresholds}")
    return float(thresholds[0])


def _acceptance_mask(scores: pd.Series, coverage: float) -> pd.Series:
    n = len(scores)
    accepted_count = min(max(int(math.ceil(float(coverage) * n)), 1), n)
    ordered = scores.sort_values(kind="mergesort").index[:accepted_count]
    accepted = pd.Series(False, index=scores.index)
    accepted.loc[ordered] = True
    return accepted


def build_failure_table(results: str | Path, output: str | Path) -> dict[str, Any]:
    results_dir = Path(results)
    for name in REQUIRED_RESULT_FILES:
        path = results_dir / name
        if not path.exists():
            raise FileNotFoundError(path)
    scenario = pd.read_csv(results_dir / "scenario_scores.csv")
    risk = pd.read_csv(results_dir / "risk_coverage.csv")
    metrics = pd.read_csv(results_dir / "model_metrics.csv")
    if scenario.empty or risk.empty or metrics.empty:
        raise ValueError("required result CSVs must be non-empty")

    threshold = _threshold_from_results(results_dir)
    coverages = sorted(float(value) for value in risk["coverage"].unique())
    judge_ids = [judge for judge in PLOT_JUDGES if f"risk_{judge}" in scenario.columns]
    rows: list[pd.DataFrame] = []
    base = scenario.rename(
        columns={
            "error": "rmse",
            "uncertainty": "uncertainty_score",
            "disagreement": "disagreement_score",
        }
    ).copy()
    base["event_error"] = np.nan
    base["bad_event_label"] = np.nan
    base["bad_rmse_label"] = base["rmse"].astype(float) > threshold
    base["combined_linear_score"] = base["risk_combined_linear"].astype(float)
    base["oracle_error_rank_score"] = base["risk_oracle_error_rank"].astype(float)
    base["random_baseline_score"] = base["risk_random_baseline"].astype(float)
    base["bad_threshold"] = threshold
    base["severity"] = "unavailable"

    for judge_id in judge_ids:
        risk_column = f"risk_{judge_id}"
        judge_base = base.copy()
        judge_base["judge_id"] = judge_id
        judge_base["risk_score"] = judge_base[risk_column].astype(float)
        for coverage in coverages:
            cov = judge_base.copy()
            cov["coverage"] = coverage
            accepted = []
            for (_, _), group in cov.groupby(["model_id", "split"], sort=False):
                accepted.append(_acceptance_mask(group["risk_score"], coverage))
            accepted_mask = pd.concat(accepted).sort_index()
            cov["accepted"] = accepted_mask.astype(bool)
            cov["false_accept"] = cov["accepted"] & cov["bad_rmse_label"].astype(bool)
            rows.append(cov)

    table = pd.concat(rows, ignore_index=True)
    required_columns = [
        "scenario_id",
        "system_id",
        "split",
        "scenario_type",
        "model_id",
        "judge_id",
        "coverage",
        "accepted",
        "risk_score",
        "false_accept",
        "rmse",
        "mae",
        "max_abs_error",
        "final_state_error",
        "event_error",
        "bad_rmse_label",
        "bad_event_label",
        "support_distance",
        "uncertainty_score",
        "disagreement_score",
        "invariant_residual",
        "repair_amount",
        "combined_linear_score",
        "oracle_error_rank_score",
        "random_baseline_score",
    ]
    missing = sorted(set(required_columns) - set(table.columns))
    if missing:
        raise ValueError(f"failure table missing required columns: {missing}")
    table = table[required_columns + [column for column in table.columns if column not in required_columns]]
    output_path = Path(output)
    _ensure_dir(output_path.parent)
    table.to_csv(output_path, index=False)

    unavailable = {
        "event_error": "not derivable from current summarized v0 artifacts; raw event trajectories are not stored",
        "bad_event_label": "not derivable from current summarized v0 artifacts; raw event trajectories are not stored",
    }
    score_nan = {
        column: bool(table[column].isna().all())
        for column in [
            "support_distance",
            "uncertainty_score",
            "disagreement_score",
            "invariant_residual",
            "repair_amount",
            "combined_linear_score",
            "oracle_error_rank_score",
            "random_baseline_score",
        ]
    }
    schema = {
        "source_results": str(results_dir),
        "output": str(output_path),
        "threshold": threshold,
        "columns": {column: str(table[column].dtype) for column in table.columns},
        "unavailable_columns": unavailable,
        "score_columns_all_nan": score_nan,
        "row_count": int(len(table)),
        "scenario_count": int(table[["scenario_id", "model_id"]].drop_duplicates().shape[0]),
    }
    _write_json(output_path.parent / "failure_table_schema.json", schema)

    verdict = "ACCEPTED" if len(table) > 0 and not any(score_nan.values()) else "REJECTED"
    report = f"""# Failure Table Report

## Input files

- {results_dir / "risk_coverage.csv"}
- {results_dir / "scenario_scores.csv"}
- {results_dir / "model_metrics.csv"}
- {results_dir / "summary.json"}

## Output files

- {output_path}
- {output_path.parent / "failure_table_schema.json"}

## Row count

{len(table)}

## Scenario count

{schema["scenario_count"]}

## Models

{", ".join(sorted(table["model_id"].unique()))}

## Judges

{", ".join(sorted(table["judge_id"].unique()))}

## Splits

{", ".join(sorted(table["split"].unique()))}

## Missing columns

{", ".join(unavailable) if unavailable else "none"}

## Numeric sanity checks

{json.dumps(score_nan, sort_keys=True)}

## Verdict

{verdict}
"""
    Path("reports/failure_table_report.md").write_text(report, encoding="utf-8")
    if verdict != "ACCEPTED":
        raise RuntimeError("failure table rejected")
    return schema


def _unique_scenario_model(table: pd.DataFrame) -> pd.DataFrame:
    return table.drop_duplicates(["scenario_id", "model_id"]).copy()


def _safe_corr(x: pd.Series, y: pd.Series, method: str) -> float:
    frame = pd.concat([x, y], axis=1).dropna()
    if len(frame) < 3:
        return np.nan
    left = frame.iloc[:, 0].astype(float)
    right = frame.iloc[:, 1].astype(float)
    if left.nunique() <= 1 or right.nunique() <= 1:
        return np.nan
    if method == "pearson":
        return float(pearsonr(left, right).statistic)
    if method == "spearman":
        return float(spearmanr(left, right).statistic)
    if method == "kendall":
        return float(kendalltau(left, right).statistic)
    raise ValueError(method)


def _safe_auc(score: pd.Series, label: pd.Series, metric: str) -> float:
    frame = pd.concat([score, label], axis=1).dropna()
    if len(frame) < 2:
        return np.nan
    y_score = frame.iloc[:, 0].astype(float)
    y_true = frame.iloc[:, 1].astype(int)
    if y_true.nunique() <= 1 or y_score.nunique() <= 1:
        return np.nan
    if metric == "auroc":
        return float(roc_auc_score(y_true, y_score))
    if metric == "auprc":
        return float(average_precision_score(y_true, y_score))
    raise ValueError(metric)


def analyze_signal_error_correlation(failure_table: str | Path, output_dir: str | Path) -> dict[str, Any]:
    table = pd.read_csv(failure_table)
    base = _unique_scenario_model(table)
    out_dir = Path(output_dir)
    _ensure_dir(out_dir)
    rows = []
    scopes = [("global", "ALL", base)]
    scopes.extend((("model", model_id, df) for model_id, df in base.groupby("model_id", sort=False)))
    for scope, model_id, df in scopes:
        for signal in SIGNALS:
            if signal not in df:
                continue
            for target in TARGETS:
                if target not in df:
                    continue
                rows.append(
                    {
                        "scope": scope,
                        "model_id": model_id,
                        "signal": signal,
                        "target": target,
                        "pearson_correlation": _safe_corr(df[signal], df[target], "pearson"),
                        "spearman_correlation": _safe_corr(df[signal], df[target], "spearman"),
                        "kendall_tau": _safe_corr(df[signal], df[target], "kendall"),
                        "auroc_for_bad_rmse": _safe_auc(df[signal], df["bad_rmse_label"], "auroc"),
                        "auprc_for_bad_rmse": _safe_auc(df[signal], df["bad_rmse_label"], "auprc"),
                        "auroc_for_bad_event": _safe_auc(df[signal], df["bad_event_label"], "auroc"),
                        "auprc_for_bad_event": _safe_auc(df[signal], df["bad_event_label"], "auprc"),
                        "n_samples": int(pd.concat([df[signal], df[target]], axis=1).dropna().shape[0]),
                    }
                )
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "signal_error_correlation.csv", index=False)
    global_rmse = result[(result["scope"] == "global") & (result["target"] == "bad_rmse_label")].copy()
    best_rmse = global_rmse.sort_values("auroc_for_bad_rmse", ascending=False, na_position="last")
    best_event = result[(result["scope"] == "global") & (result["target"] == "bad_event_label")].sort_values(
        "auroc_for_bad_event", ascending=False, na_position="last"
    )
    best_auc = float(best_rmse["auroc_for_bad_rmse"].dropna().max()) if not best_rmse["auroc_for_bad_rmse"].dropna().empty else np.nan
    if pd.isna(best_auc) or best_auc <= 0.55:
        verdict = "NO_SIGNAL"
    elif best_auc < 0.70:
        verdict = "WEAK_SIGNALS"
    else:
        verdict = "USEFUL_SIGNALS_FOUND"
    near_random_mask = best_rmse["auroc_for_bad_rmse"].between(0.45, 0.55, inclusive="both").fillna(False)
    near_random = best_rmse[near_random_mask]["signal"].tolist()
    negative = result[(result["scope"] == "global") & (result["spearman_correlation"] < 0)]["signal"].drop_duplicates().tolist()
    summary = {
        "verdict": verdict,
        "best_bad_rmse_signal": None if best_rmse.empty else str(best_rmse.iloc[0]["signal"]),
        "best_bad_rmse_auroc": None if pd.isna(best_auc) else best_auc,
        "event_metrics_available": bool(best_event["auroc_for_bad_event"].notna().any()) if not best_event.empty else False,
        "near_random_signals": near_random,
        "negative_correlated_signals": negative,
        "key_finding": f"Best RMSE-failure AUROC was {best_auc:.6f}" if not pd.isna(best_auc) else "AUROC unavailable",
    }
    _write_json(out_dir / "signal_error_correlation_summary.json", summary)
    report = f"""# Signal-Error Correlation Report

## Question

Do validator signals predict actual model failure?

## Signals analyzed

{", ".join(SIGNALS)}

## Targets analyzed

{", ".join(TARGETS)}

## Best signals for RMSE failure

{_markdown_table(best_rmse.rename(columns={"auroc_for_bad_rmse": "auroc", "auprc_for_bad_rmse": "auprc", "spearman_correlation": "spearman", "n_samples": "n"}), ["signal", "auroc", "auprc", "spearman", "n"], max_rows=8)}

## Best signals for event failure

{_markdown_table(best_event.rename(columns={"auroc_for_bad_event": "auroc", "auprc_for_bad_event": "auprc", "spearman_correlation": "spearman", "n_samples": "n"}), ["signal", "auroc", "auprc", "spearman", "n"], max_rows=8)}

## Signals that are near random

{", ".join(near_random) if near_random else "none"}

## Signals that are negatively correlated

{", ".join(negative) if negative else "none"}

## Interpretation

{summary["key_finding"]}. Event labels are unavailable in current v0 artifacts, so event AUROC/AUPRC are not fabricated.

## Verdict

{verdict}
"""
    Path("reports/signal_error_correlation.md").write_text(report, encoding="utf-8")
    return summary


def _far_by_group(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return (
        df.groupby(group_cols, as_index=False)
        .agg(
            scenario_count=("scenario_id", "nunique"),
            mean_rmse=("rmse", "mean"),
            median_rmse=("rmse", "median"),
            bad_rmse_rate=("bad_rmse_label", "mean"),
            event_failure_rate=("bad_event_label", "mean"),
            false_accept_rate=("false_accept", "mean"),
            accepted_count=("accepted", "sum"),
            false_accept_count=("false_accept", "sum"),
        )
    )


def analyze_per_split_failure(failure_table: str | Path, output_dir: str | Path) -> dict[str, Any]:
    table = pd.read_csv(failure_table)
    out_dir = Path(output_dir)
    _ensure_dir(out_dir)
    split_detail = _far_by_group(table, ["split", "scenario_type", "model_id", "judge_id", "coverage"])
    split_detail.to_csv(out_dir / "per_split_failure.csv", index=False)
    unique = _unique_scenario_model(table)
    error_by_split = (
        unique.groupby(["split", "scenario_type"], as_index=False)
        .agg(mean_rmse=("rmse", "mean"), bad_rmse_rate=("bad_rmse_label", "mean"))
        .sort_values("mean_rmse", ascending=False)
    )
    judge_split = (
        table[table["judge_id"].isin(SIMPLE_JUDGES + ["combined_linear"])]
        .groupby(["split", "judge_id"], as_index=False)["false_accept"]
        .mean()
        .rename(columns={"false_accept": "false_accept_rate"})
    )
    best_rows = []
    for split, group in judge_split.groupby("split", sort=False):
        simple = group[group["judge_id"].isin(SIMPLE_JUDGES)].sort_values("false_accept_rate")
        combined = float(group.loc[group["judge_id"] == "combined_linear", "false_accept_rate"].iloc[0])
        best = simple.iloc[0]
        best_rows.append(
            {
                "split": split,
                "best_real_judge": best["judge_id"],
                "false_accept_rate": float(best["false_accept_rate"]),
                "combined_linear_far": combined,
                "combined_margin": float(best["false_accept_rate"]) - combined,
                "combined_wins": combined < float(best["false_accept_rate"]),
            }
        )
    best_df = pd.DataFrame(best_rows)
    bad_ood = unique[unique["split"] != "id_test"]["bad_rmse_label"].mean()
    split_false_accepts = table[table["split"] != "id_test"].groupby("split")["false_accept"].sum().sort_values(ascending=False)
    dominates = bool(not split_false_accepts.empty and split_false_accepts.iloc[0] >= max(1, split_false_accepts.sum() * 0.5))
    if bad_ood < 0.05:
        verdict = "BENCHMARK_TOO_EASY"
    elif dominates and len(split_false_accepts) <= 2:
        verdict = "SPLIT_SPECIFIC_FAILURE"
    else:
        verdict = "GLOBAL_FAILURE"
    failed = best_df[best_df["combined_margin"] <= 0].sort_values("combined_margin")
    worked = best_df[best_df["combined_margin"] > 0].sort_values("combined_margin", ascending=False)
    summary = {
        "verdict": verdict,
        "worst_split_by_error": str(error_by_split.iloc[0]["split"]),
        "worst_split_mean_rmse": float(error_by_split.iloc[0]["mean_rmse"]),
        "combined_failed_splits": failed["split"].tolist(),
        "combined_worked_splits": worked["split"].tolist(),
        "key_finding": f"Worst split by mean RMSE: {error_by_split.iloc[0]['split']}",
    }
    _write_json(out_dir / "per_split_summary.json", summary)
    worst_fa = split_detail.sort_values("false_accept_rate", ascending=False).rename(columns={"judge_id": "judge"})
    report = f"""# Per-Split Failure Analysis

## Worst splits by model error

{_markdown_table(error_by_split, ["split", "scenario_type", "mean_rmse", "bad_rmse_rate"], max_rows=10)}

## Worst splits by false accepts

{_markdown_table(worst_fa, ["split", "scenario_type", "judge", "coverage", "false_accept_rate"], max_rows=12)}

## Best real judge per split

{_markdown_table(best_df, ["split", "best_real_judge", "false_accept_rate", "combined_linear_far", "combined_margin"])}

## Where combined_linear failed

{", ".join(failed["split"].tolist()) if not failed.empty else "none"}

## Where combined_linear worked

{", ".join(worked["split"].tolist()) if not worked.empty else "none"}

## Interpretation

{summary["key_finding"]}; combined failures are listed above.

## Verdict

{verdict}
"""
    Path("reports/per_split_failure.md").write_text(report, encoding="utf-8")
    return summary


def _recompute_far_from_scores(
    table: pd.DataFrame,
    threshold: float,
    coverages: list[float],
    score_column_by_judge: dict[str, str],
) -> pd.DataFrame:
    base = _unique_scenario_model(table)
    rows = []
    for model_id, split in base[["model_id", "split"]].drop_duplicates().itertuples(index=False):
        group = base[(base["model_id"] == model_id) & (base["split"] == split)].copy()
        bad = group["rmse"].astype(float) > threshold
        for judge_id, score_col in score_column_by_judge.items():
            scores = group[score_col].astype(float)
            for coverage in coverages:
                accepted = _acceptance_mask(scores, coverage)
                accepted_count = int(accepted.sum())
                false_accept_count = int((accepted & bad).sum())
                rows.append(
                    {
                        "model_id": model_id,
                        "split": split,
                        "judge_id": judge_id,
                        "coverage": float(coverage),
                        "threshold": float(threshold),
                        "bad_rate": float(bad.mean()),
                        "false_accept_rate": float(false_accept_count / accepted_count) if accepted_count else np.nan,
                        "accepted_count": accepted_count,
                        "false_accept_count": false_accept_count,
                        "available": bool(bad.nunique() > 1),
                    }
                )
    return pd.DataFrame(rows)


def _claim_from_far(far: pd.DataFrame) -> tuple[float, str, str]:
    rows = []
    for (model_id, split, coverage), group in far.groupby(["model_id", "split", "coverage"], sort=False):
        combined = group[group["judge_id"] == "combined_linear"]
        simple = group[group["judge_id"].isin(SIMPLE_JUDGES)]
        if combined.empty or simple.empty:
            continue
        best = simple.sort_values("false_accept_rate").iloc[0]
        rows.append(float(combined.iloc[0]["false_accept_rate"]) < float(best["false_accept_rate"]))
    win_rate = float(np.mean(rows)) if rows else 0.0
    strongest = str(far[far["judge_id"].isin(SIMPLE_JUDGES)].groupby("judge_id")["false_accept_rate"].mean().sort_values().index[0])
    verdict = "SUPPORTED" if win_rate >= 0.70 else ("MIXED" if win_rate >= 0.40 else "NOT_SUPPORTED")
    return win_rate, strongest, verdict


def analyze_threshold_sensitivity(failure_table: str | Path, thresholds: list[float], output_dir: str | Path) -> dict[str, Any]:
    table = pd.read_csv(failure_table)
    out_dir = Path(output_dir)
    _ensure_dir(out_dir)
    score_cols = {
        "support_only": "support_distance",
        "uncertainty_only": "uncertainty_score",
        "disagreement_only": "disagreement_score",
        "invariant_only": "invariant_residual",
        "repair_only": "repair_amount",
        "combined_linear": "combined_linear_score",
        "random_baseline": "random_baseline_score",
        "oracle_error_rank": "oracle_error_rank_score",
    }
    rows = []
    coverages = sorted(table["coverage"].unique())
    base = table.drop_duplicates(["scenario_id", "model_id"])
    for threshold in thresholds:
        far = _recompute_far_from_scores(table, threshold, coverages, score_cols)
        bad_rate = float((base["rmse"] > threshold).mean())
        if bad_rate in {0.0, 1.0}:
            win_rate, strongest, verdict = 0.0, "UNAVAILABLE", "UNAVAILABLE"
        else:
            win_rate, strongest, verdict = _claim_from_far(far)
        for _, row in far.iterrows():
            rows.append({**row.to_dict(), "combined_win_rate": win_rate, "strongest_simple_judge": strongest, "verdict": verdict})
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "threshold_sensitivity.csv", index=False)
    summary_df = (
        result.groupby("threshold", as_index=False)
        .agg(
            bad_rate=("bad_rate", "mean"),
            combined_win_rate=("combined_win_rate", "first"),
            strongest_simple_judge=("strongest_simple_judge", "first"),
            verdict=("verdict", "first"),
        )
    )
    works = summary_df[summary_df["combined_win_rate"] >= 0.70]["threshold"].tolist()
    if len(works) >= math.ceil(0.7 * len(thresholds)):
        overall = "ROBUST_TO_THRESHOLD"
    elif works:
        overall = "THRESHOLD_DEPENDENT"
    else:
        overall = "UNSUPPORTED_ACROSS_THRESHOLDS"
    summary = {
        "verdict": overall,
        "thresholds": thresholds,
        "thresholds_where_combined_works": works,
        "key_finding": f"combined_linear worked at {len(works)} of {len(thresholds)} thresholds",
    }
    _write_json(out_dir / "threshold_sensitivity_summary.json", summary)
    report = f"""# Threshold Sensitivity Report

## Thresholds tested

{", ".join(str(value) for value in thresholds)}

## Bad scenario rate by threshold

{_markdown_table(summary_df, ["threshold", "bad_rate"])}

## Combined judge result by threshold

{_markdown_table(summary_df, ["threshold", "combined_win_rate", "strongest_simple_judge", "verdict"])}

## Thresholds where combined works

{", ".join(map(str, works)) if works else "none"}

## Thresholds where combined fails

{", ".join(map(str, summary_df[summary_df["combined_win_rate"] < 0.70]["threshold"].tolist()))}

## Interpretation

{summary["key_finding"]}. Thresholds with degenerate labels are marked unavailable, not counted as wins.

## Verdict

{overall}
"""
    Path("reports/threshold_sensitivity.md").write_text(report, encoding="utf-8")
    return summary


def analyze_coverage_sensitivity(failure_table: str | Path, coverages: list[float], output_dir: str | Path) -> dict[str, Any]:
    table = pd.read_csv(failure_table)
    out_dir = Path(output_dir)
    _ensure_dir(out_dir)
    threshold = float(table["bad_threshold"].iloc[0]) if "bad_threshold" in table else float(table["rmse"].median())
    score_cols = {
        "support_only": "support_distance",
        "uncertainty_only": "uncertainty_score",
        "disagreement_only": "disagreement_score",
        "invariant_only": "invariant_residual",
        "repair_only": "repair_amount",
        "combined_linear": "combined_linear_score",
        "random_baseline": "random_baseline_score",
        "oracle_error_rank": "oracle_error_rank_score",
    }
    far = _recompute_far_from_scores(table, threshold, coverages, score_cols)
    far.to_csv(out_dir / "coverage_sensitivity.csv", index=False)
    rows = []
    for coverage, group in far.groupby("coverage", sort=True):
        simple = group[group["judge_id"].isin(SIMPLE_JUDGES)].groupby("judge_id")["false_accept_rate"].mean().sort_values()
        combined = float(group[group["judge_id"] == "combined_linear"]["false_accept_rate"].mean())
        oracle = float(group[group["judge_id"] == "oracle_error_rank"]["false_accept_rate"].mean())
        best_judge = str(simple.index[0])
        best_far = float(simple.iloc[0])
        rows.append(
            {
                "coverage": float(coverage),
                "best_real_judge": best_judge,
                "best_real_far": best_far,
                "combined_far": combined,
                "combined_margin": best_far - combined,
                "oracle_far": oracle,
                "oracle_gap": best_far - oracle,
            }
        )
    best_df = pd.DataFrame(rows)
    works = best_df[best_df["combined_margin"] > 0]["coverage"].tolist()
    if works and max(works) <= 0.30:
        verdict = "WORKS_AT_LOW_COVERAGE"
    elif works and min(works) >= 0.70:
        verdict = "WORKS_AT_HIGH_COVERAGE"
    elif not works:
        verdict = "FAILS_ACROSS_COVERAGE"
    else:
        verdict = "MIXED"
    summary = {
        "verdict": verdict,
        "coverage_where_combined_works": works,
        "coverage_where_combined_fails": best_df[best_df["combined_margin"] <= 0]["coverage"].tolist(),
        "oracle_gap_mean": float(best_df["oracle_gap"].mean()),
        "key_finding": f"combined_linear worked at {len(works)} coverage points",
    }
    _write_json(out_dir / "coverage_sensitivity_summary.json", summary)
    plot_df = far.groupby(["coverage", "judge_id"], as_index=False)["false_accept_rate"].mean()
    plt.figure(figsize=(8, 5))
    for judge in PLOT_JUDGES:
        judge_df = plot_df[plot_df["judge_id"] == judge]
        if judge_df.empty:
            continue
        label = f"{judge} (diagnostic)" if judge == "oracle_error_rank" else judge
        plt.plot(judge_df["coverage"], judge_df["false_accept_rate"], marker="o", label=label)
    plt.xlabel("coverage")
    plt.ylabel("false_accept_rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "coverage_sensitivity.png")
    plt.close()
    report = f"""# Coverage Sensitivity Report

## Coverage grid

{", ".join(str(value) for value in coverages)}

## Best judge by coverage

{_markdown_table(best_df, ["coverage", "best_real_judge", "best_real_far", "combined_far", "combined_margin"])}

## Coverage regions where combined works

{", ".join(map(str, works)) if works else "none"}

## Coverage regions where combined fails

{", ".join(map(str, summary["coverage_where_combined_fails"]))}

## Oracle gap

Mean oracle gap: {summary["oracle_gap_mean"]:.6f}. Oracle is diagnostic only.

## Interpretation

{summary["key_finding"]}.

## Verdict

{verdict}
"""
    Path("reports/coverage_sensitivity.md").write_text(report, encoding="utf-8")
    return summary


def _normalize(values: pd.Series) -> pd.Series:
    values = values.astype(float)
    denom = values.max() - values.min()
    if denom <= 1e-12:
        return pd.Series(0.0, index=values.index)
    return (values - values.min()) / denom


def _candidate_score_frame(base: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    scores = pd.DataFrame(index=base.index)
    normalized = {signal: _normalize(base[signal]) for signal in BASE_SIGNALS}
    scores["combined_linear"] = base["combined_linear_score"].astype(float)
    for signal in BASE_SIGNALS:
        remaining = [normalized[item] for item in BASE_SIGNALS if item != signal]
        scores[f"combined_without_{signal.replace('_score', '').replace('_distance', '').replace('_residual', '').replace('_amount', '')}"] = sum(remaining) / len(remaining)
    ranks = [base[signal].rank(method="average", pct=True) for signal in BASE_SIGNALS]
    scores["rank_normalized_combined"] = sum(ranks) / len(ranks)
    caveats = {"rank_normalized_combined": "rank-normalized deterministic diagnostic"}
    return scores, caveats


def _cross_validated_scores(base: pd.DataFrame) -> tuple[pd.Series, pd.Series, str]:
    features = BASE_SIGNALS
    y = base["bad_rmse_label"].astype(int)
    groups = base["scenario_id"].astype(str)
    if y.nunique() < 2 or groups.nunique() < 4:
        nan = pd.Series(np.nan, index=base.index)
        return nan, nan, "insufficient non-degenerate grouped data"
    splits = min(3, groups.nunique())
    logistic_scores = pd.Series(np.nan, index=base.index, dtype=float)
    isotonic_scores = pd.Series(np.nan, index=base.index, dtype=float)
    splitter = GroupKFold(n_splits=splits)
    for train_idx, test_idx in splitter.split(base[features], y, groups):
        train = base.iloc[train_idx]
        test = base.iloc[test_idx]
        if train["bad_rmse_label"].nunique() < 2:
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
        model.fit(train[features], train["bad_rmse_label"].astype(int))
        logistic_scores.iloc[test_idx] = model.predict_proba(test[features])[:, 1]

        group_split = GroupShuffleSplit(n_splits=1, test_size=0.35, random_state=17)
        sub_train_idx, cal_idx = next(group_split.split(train[features], train["bad_rmse_label"], train["scenario_id"]))
        sub_train = train.iloc[sub_train_idx]
        cal = train.iloc[cal_idx]
        if sub_train["bad_rmse_label"].nunique() < 2 or cal["bad_rmse_label"].nunique() < 2:
            continue
        calibrated_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
        calibrated_model.fit(sub_train[features], sub_train["bad_rmse_label"].astype(int))
        cal_score = calibrated_model.predict_proba(cal[features])[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(cal_score, cal["bad_rmse_label"].astype(int))
        isotonic_scores.iloc[test_idx] = iso.predict(calibrated_model.predict_proba(test[features])[:, 1])
    return logistic_scores, isotonic_scores, "grouped cross-validation by scenario_id"


def analyze_score_ablation(failure_table: str | Path, output_dir: str | Path) -> dict[str, Any]:
    table = pd.read_csv(failure_table)
    out_dir = Path(output_dir)
    _ensure_dir(out_dir)
    base = _unique_scenario_model(table)
    scores, caveats = _candidate_score_frame(base)
    logistic, isotonic, validation_scheme = _cross_validated_scores(base)
    scores["logistic_error_classifier"] = logistic
    scores["isotonic_calibrated_score"] = isotonic
    caveats["logistic_error_classifier"] = validation_scheme
    caveats["isotonic_calibrated_score"] = validation_scheme
    coverages = sorted(table["coverage"].unique())
    rows = []
    score_table = base.copy()
    for score_name in scores.columns:
        score_table[score_name] = scores[score_name]
    for score_name in scores.columns:
        if score_table[score_name].isna().all():
            rows.append({"score": score_name, "false_accept_rate": np.nan, "win_rate_vs_best_simple": 0.0, "verdict": "UNAVAILABLE"})
            continue
        temp = table.copy()
        mapping = score_table.set_index(["scenario_id", "model_id"])[score_name]
        temp["candidate_score"] = [mapping.loc[(row.scenario_id, row.model_id)] for row in temp.itertuples()]
        threshold = float(temp["bad_threshold"].iloc[0]) if "bad_threshold" in temp else float(temp["rmse"][temp["bad_rmse_label"].astype(bool)].min())
        far = _recompute_far_from_scores(temp, threshold, coverages, {"candidate": "candidate_score", **{
            "support_only": "support_distance",
            "uncertainty_only": "uncertainty_score",
            "disagreement_only": "disagreement_score",
            "invariant_only": "invariant_residual",
            "repair_only": "repair_amount",
            "random_baseline": "random_baseline_score",
        }})
        candidate = far[far["judge_id"] == "candidate"]["false_accept_rate"].mean()
        compare_rows = []
        for (model_id, split, coverage), group in far.groupby(["model_id", "split", "coverage"]):
            cand = float(group[group["judge_id"] == "candidate"]["false_accept_rate"].iloc[0])
            best = float(group[group["judge_id"].isin(SIMPLE_JUDGES)]["false_accept_rate"].min())
            compare_rows.append(cand < best)
        win_rate = float(np.mean(compare_rows)) if compare_rows else 0.0
        verdict = "SUPPORTED" if win_rate >= 0.70 else ("MIXED" if win_rate >= 0.40 else "NOT_SUPPORTED")
        rows.append({"score": score_name, "false_accept_rate": float(candidate), "win_rate_vs_best_simple": win_rate, "verdict": verdict})
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "score_ablation.csv", index=False)
    combined_far = float(result.loc[result["score"] == "combined_linear", "false_accept_rate"].iloc[0])
    removal = result[result["score"].str.startswith("combined_without")].copy()
    removal["removed_signal"] = removal["score"].str.replace("combined_without_", "", regex=False)
    removal["delta_vs_combined_linear"] = removal["false_accept_rate"] - combined_far
    best_win = float(result["win_rate_vs_best_simple"].max())
    if best_win >= 0.70 and best_win > float(result.loc[result["score"] == "combined_linear", "win_rate_vs_best_simple"].iloc[0]):
        verdict = "COMBINATION_PROBLEM"
    elif best_win < 0.40:
        verdict = "SIGNAL_PROBLEM"
    else:
        verdict = "INCONCLUSIVE"
    summary = {
        "verdict": verdict,
        "best_score": str(result.sort_values("win_rate_vs_best_simple", ascending=False).iloc[0]["score"]),
        "best_win_rate": best_win,
        "validation_scheme": validation_scheme,
        "key_finding": f"Best ablation/calibration win rate was {best_win:.6f}",
    }
    _write_json(out_dir / "score_ablation_summary.json", summary)
    learned = result[result["score"].isin(["logistic_error_classifier", "isotonic_calibrated_score"])].copy()
    learned["method"] = learned["score"]
    learned["validation_scheme"] = validation_scheme
    learned["win_rate"] = learned["win_rate_vs_best_simple"]
    learned["caveat"] = learned["method"].map(caveats).fillna("")
    report = f"""# Score Ablation and Calibration Report

## Question

Did combined_linear fail because of bad combination logic?

## Ablation results

{_markdown_table(result, ["score", "false_accept_rate", "win_rate_vs_best_simple", "verdict"])}

## Signal removal effect

{_markdown_table(removal, ["removed_signal", "delta_vs_combined_linear"])}

## Learned calibration result

{_markdown_table(learned, ["method", "validation_scheme", "win_rate", "caveat"])}

## Signals that hurt the combined score

{", ".join(removal[removal["delta_vs_combined_linear"] < 0]["removed_signal"].tolist()) or "none"}

## Signals that help the combined score

{", ".join(removal[removal["delta_vs_combined_linear"] > 0]["removed_signal"].tolist()) or "none"}

## Interpretation

{summary["key_finding"]}; learned scores used grouped validation, not same-row train/test evaluation.

## Verdict

{verdict}
"""
    Path("reports/score_ablation.md").write_text(report, encoding="utf-8")
    return summary


def analyze_model_diversity(failure_table: str | Path, output_dir: str | Path) -> dict[str, Any]:
    table = pd.read_csv(failure_table)
    out_dir = Path(output_dir)
    _ensure_dir(out_dir)
    base = _unique_scenario_model(table)
    model_rows = []
    for split, group in base.groupby("split", sort=False):
        by_model = group.groupby("model_id")["rmse"].mean().sort_values()
        model_rows.append(
            {
                "split": split,
                "best_model": str(by_model.index[0]),
                "worst_model": str(by_model.index[-1]),
                "error_gap": float(by_model.iloc[-1] - by_model.iloc[0]),
                "mean_pairwise_disagreement": float(group["disagreement_score"].mean()),
            }
        )
    model_df = pd.DataFrame(model_rows)
    model_df.to_csv(out_dir / "model_diversity.csv", index=False)
    oracle_rows = []
    for split, group in table.groupby("split", sort=False):
        far = group.groupby("judge_id")["false_accept"].mean()
        oracle_far = float(far.get("oracle_error_rank", np.nan))
        best_real = float(far[far.index.isin(REAL_JUDGES)].min())
        oracle_rows.append({"split": split, "oracle_far": oracle_far, "best_real_judge_far": best_real, "oracle_gap": best_real - oracle_far})
    oracle_df = pd.DataFrame(oracle_rows)
    oracle_df.to_csv(out_dir / "oracle_gap.csv", index=False)
    mean_disagreement = float(model_df["mean_pairwise_disagreement"].mean())
    mean_error_gap = float(model_df["error_gap"].mean())
    mean_bad_rate = float(base["bad_rmse_label"].mean())
    mean_oracle_gap = float(oracle_df["oracle_gap"].mean())
    if mean_disagreement < 0.05 and mean_error_gap < 0.05:
        verdict = "MODELS_TOO_SIMILAR"
    elif mean_bad_rate > 0.80:
        verdict = "MODELS_TOO_BAD"
    elif mean_oracle_gap > 0.10:
        verdict = "REAL_SIGNALS_MISSING"
    else:
        verdict = "ORACLE_GAP_SMALL"
    summary = {
        "verdict": verdict,
        "mean_pairwise_prediction_disagreement": mean_disagreement,
        "mean_error_gap": mean_error_gap,
        "mean_oracle_gap": mean_oracle_gap,
        "key_finding": f"Mean oracle gap was {mean_oracle_gap:.6f}",
    }
    _write_json(out_dir / "model_diversity_summary.json", summary)
    report = f"""# Model Diversity and Oracle Gap Report

## Model error ranking

{_markdown_table(model_df, ["split", "best_model", "worst_model", "error_gap"])}

## Model disagreement

{_markdown_table(model_df, ["split", "mean_pairwise_disagreement"])}

## Oracle gap

{_markdown_table(oracle_df, ["split", "oracle_far", "best_real_judge_far", "oracle_gap"])}

## Interpretation

{summary["key_finding"]}. Oracle is diagnostic and not deployable.

## Verdict

{verdict}
"""
    Path("reports/model_diversity_and_oracle_gap.md").write_text(report, encoding="utf-8")
    return summary


def analyze_benchmark_sanity(results: str | Path, failure_table: str | Path, output_dir: str | Path) -> dict[str, Any]:
    results_dir = Path(results)
    table = pd.read_csv(failure_table)
    out_dir = Path(output_dir)
    _ensure_dir(out_dir)
    summary = _load_json(results_dir / "summary.json")
    data_summary = summary.get("dataset_summary", {})
    checks = []
    train = data_summary.get("train", {})
    train_action_range = float(train.get("action_max", 0.0)) - float(train.get("action_min", 0.0))
    for split in ["ood_action_magnitude", "ood_inflow_spike", "ood_combined"]:
        item = data_summary.get(split, {})
        if not item:
            continue
        action_range = float(item.get("action_max", 0.0)) - float(item.get("action_min", 0.0))
        checks.append(
            {
                "check": f"{split}_action_range_ratio",
                "passed": bool(action_range > train_action_range * 1.25),
                "value": action_range / train_action_range if train_action_range else np.nan,
                "threshold": 1.25,
            }
        )
        if "disturbance_0_max" in item and "disturbance_0_max" in train:
            ratio = float(item["disturbance_0_max"]) / max(float(train["disturbance_0_max"]), 1e-12)
            checks.append(
                {
                    "check": f"{split}_disturbance0_max_ratio",
                    "passed": bool(ratio > 1.25),
                    "value": ratio,
                    "threshold": 1.25,
                }
            )
    checks_df = pd.DataFrame(checks)
    checks_df.to_csv(out_dir / "benchmark_sanity.csv", index=False)
    unique = _unique_scenario_model(table)
    error_sep = unique.groupby("split", as_index=False).agg(mean_error=("rmse", "mean"), bad_rate=("bad_rmse_label", "mean"))
    id_error = float(error_sep.loc[error_sep["split"] == "id_test", "mean_error"].iloc[0])
    ood_error = float(error_sep.loc[error_sep["split"] != "id_test", "mean_error"].mean())
    bad_rate = float(unique["bad_rmse_label"].mean())
    nondegenerate_bad = 0.0 < bad_rate < 1.0
    event_available = bool(unique["bad_event_label"].notna().any())
    enough_bad = int(unique["bad_rmse_label"].sum()) >= 10
    accepted_bad = int(table["false_accept"].sum()) >= 10
    ood_diff = bool(checks_df["passed"].any()) if not checks_df.empty else False
    error_separated = bool(ood_error > id_error * 1.10)
    if ood_diff and error_separated and nondegenerate_bad:
        verdict = "VALID_BENCHMARK"
    elif ood_diff:
        verdict = "WEAK_BENCHMARK"
    else:
        verdict = "INVALID_BENCHMARK"
    result = {
        "verdict": verdict,
        "ood_distribution_differs": ood_diff,
        "id_mean_error": id_error,
        "ood_mean_error": ood_error,
        "bad_label_rate": bad_rate,
        "bad_labels_nondegenerate": nondegenerate_bad,
        "event_labels_available": event_available,
        "enough_bad_scenarios": enough_bad,
        "enough_accepted_bad_scenarios": accepted_bad,
        "key_finding": f"OOD mean error {ood_error:.6f} vs ID mean error {id_error:.6f}",
    }
    _write_json(out_dir / "benchmark_sanity_summary.json", result)
    report = f"""# Benchmark Sanity Report

## OOD distribution checks

{_markdown_table(checks_df, ["check", "passed", "value", "threshold"])}

## Error separation

{_markdown_table(error_sep, ["split", "mean_error", "bad_rate"])}

## Label degeneracy

Bad label rate: {bad_rate:.6f}; non-degenerate: {nondegenerate_bad}; enough bad scenarios: {enough_bad}; enough accepted bad scenarios: {accepted_bad}.

## Event degeneracy

Event labels available: {event_available}. Current v0 artifacts do not store raw event trajectories.

## Benchmark verdict

{verdict}

## Explanation

{result["key_finding"]}.

## Required fixes if weak or invalid

{"- none" if verdict == "VALID_BENCHMARK" else "- Improve error separation and store event trajectories for event-label analysis."}
"""
    Path("reports/benchmark_sanity.md").write_text(report, encoding="utf-8")
    return result


def make_failure_diagnosis(input_dir: str | Path, output: str | Path) -> dict[str, Any]:
    source = Path(input_dir)
    required = {
        "gate": "gate_verification.json",
        "table": "failure_table_schema.json",
        "signal": "signal_error_correlation_summary.json",
        "split": "per_split_summary.json",
        "threshold": "threshold_sensitivity_summary.json",
        "coverage": "coverage_sensitivity_summary.json",
        "ablation": "score_ablation_summary.json",
        "model": "model_diversity_summary.json",
        "benchmark": "benchmark_sanity_summary.json",
    }
    loaded = {}
    missing = []
    for key, filename in required.items():
        path = source / filename
        if not path.exists():
            missing.append(str(path))
        else:
            loaded[key] = _load_json(path)
    if missing:
        raise FileNotFoundError(", ".join(missing))

    benchmark_v = loaded["benchmark"]["verdict"]
    signal_v = loaded["signal"]["verdict"]
    ablation_v = loaded["ablation"]["verdict"]
    model_v = loaded["model"]["verdict"]
    threshold_v = loaded["threshold"]["verdict"]
    coverage_v = loaded["coverage"]["verdict"]
    oracle_gap_small = model_v == "ORACLE_GAP_SMALL"
    combined_failed_splits = loaded["split"].get("combined_failed_splits", [])
    if benchmark_v in {"WEAK_BENCHMARK", "INVALID_BENCHMARK"}:
        diagnosis = "BENCHMARK_PROBLEM"
        action = "FIX_BENCHMARK"
    elif signal_v == "USEFUL_SIGNALS_FOUND" and (
        ablation_v == "COMBINATION_PROBLEM"
        or threshold_v == "UNSUPPORTED_ACROSS_THRESHOLDS"
        or coverage_v in {"FAILS_ACROSS_COVERAGE", "WORKS_AT_LOW_COVERAGE"}
        or bool(combined_failed_splits)
    ):
        diagnosis = "JUDGE_PROBLEM"
        action = "REPLACE_JUDGE"
    elif model_v in {"MODELS_TOO_SIMILAR", "MODELS_TOO_BAD"}:
        diagnosis = "MODEL_PROBLEM"
        action = "IMPROVE_MODELS"
    elif (
        benchmark_v == "VALID_BENCHMARK"
        and signal_v in {"WEAK_SIGNALS", "NO_SIGNAL"}
        and ablation_v != "COMBINATION_PROBLEM"
        and threshold_v == "UNSUPPORTED_ACROSS_THRESHOLDS"
        and coverage_v == "FAILS_ACROSS_COVERAGE"
        and oracle_gap_small
    ):
        diagnosis = "IDEA_NOT_SUPPORTED"
        action = "KILL_OR_DOWNGRADE_CLAIM"
    elif benchmark_v == "VALID_BENCHMARK" and signal_v in {"WEAK_SIGNALS", "NO_SIGNAL"}:
        diagnosis = "SIGNAL_PROBLEM"
        action = "RUN_MINIMAL_ADDITIONAL_TEST"
    else:
        diagnosis = "INCONCLUSIVE"
        action = "RUN_MINIMAL_ADDITIONAL_TEST"

    rows = [
        {"analysis": "gate", "verdict": loaded["gate"]["verdict"], "key_finding": loaded["gate"]["expansion_status"]},
        {"analysis": "signal_error_correlation", "verdict": signal_v, "key_finding": loaded["signal"]["key_finding"]},
        {"analysis": "per_split_failure", "verdict": loaded["split"]["verdict"], "key_finding": loaded["split"]["key_finding"]},
        {"analysis": "threshold_sensitivity", "verdict": threshold_v, "key_finding": loaded["threshold"]["key_finding"]},
        {"analysis": "coverage_sensitivity", "verdict": coverage_v, "key_finding": loaded["coverage"]["key_finding"]},
        {"analysis": "score_ablation", "verdict": ablation_v, "key_finding": loaded["ablation"]["key_finding"]},
        {"analysis": "model_diversity", "verdict": model_v, "key_finding": loaded["model"]["key_finding"]},
        {"analysis": "benchmark_sanity", "verdict": benchmark_v, "key_finding": loaded["benchmark"]["key_finding"]},
    ]
    report = {
        "diagnosis": diagnosis,
        "recommended_next_action": action,
        "evidence": rows,
        "expansion_forbidden": action not in {"RUN_MINIMAL_ADDITIONAL_TEST"} or loaded["gate"]["expansion_status"] == "BLOCKED",
    }
    output_path = Path(output)
    _ensure_dir(output_path.parent)
    _write_json(output_path.with_suffix(".json"), report)
    evidence_df = pd.DataFrame(rows)
    text = f"""# Failure Diagnosis

## Starting point

The v0 decision gate did not allow expansion. The original combined_linear claim is not supported.

## Evidence summary

{_markdown_table(evidence_df, ["analysis", "verdict", "key_finding"])}

## Diagnosis

{diagnosis}

## Explanation

The diagnosis follows the rule hierarchy in the failure-analysis task. Expansion remains blocked by the v0 gate unless a future minimal test changes the diagnosis.

## What not to do next

- Do not add CSTR, RSSM, or new systems as a way to rescue the failed v0 claim.
- Do not treat oracle_error_rank as a deployable judge.
- Do not call combined_linear supported without rerunning this diagnosis after a real fix.

## Recommended next action

{action}

## Concrete next milestone

Run one minimal fix aligned with the diagnosis, then rerun the full failure-analysis command chain.

## Claims allowed after this analysis

- The current v0 claim is unsupported.
- The listed diagnostics identify the current failure mode for this setup.

## Claims still forbidden

- combined_linear is robustly better than the strongest simple judge.
- Expansion results validate the original claim.
"""
    output_path.write_text(text, encoding="utf-8")
    return report
