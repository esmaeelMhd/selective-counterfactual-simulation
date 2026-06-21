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
from scipy import stats

from scs.data.schemas import load_dataset
from scs.experiments.registry import make_model


BASELINE_JUDGE = "best_single_signal_selected_on_calibration"
CALIBRATED_FAMILY = [
    "calibration_selected_candidate_ranker",
    "rank_normalized_linear",
    "logistic_calibrated_judge",
    "isotonic_calibrated_judge",
    "quantile_rule_judge",
    "conservative_low_coverage_judge",
]
DEPLOYABLE_SIGNALS = [
    "support_distance",
    "uncertainty_score",
    "disagreement_score",
    "invariant_residual",
    "repair_amount",
]
STATEWISE_COLUMNS = [
    "state_error_concentration_rmse",
    "state_error_temperature_rmse",
    "state_error_concentration_max_abs",
    "state_error_temperature_max_abs",
    "state_error_concentration_final",
    "state_error_temperature_final",
]
OLD_REPO_NAMES = [
    "time" + "-series" + "-simulator",
    "digital" + "-twin" + "-engine",
    "flux" + "-attention" + "-engine",
    "plant" + "-scenario" + "-compiler",
]
PRIOR_ARTIFACTS = [
    "results/calibrated_cstr/calibrated_risk_coverage.csv",
    "results/calibrated_cstr/test_table.csv",
    "results/calibrated_cstr/calibrated_judge_summary.json",
    "results/calibrated_cstr/low_coverage_summary.csv",
    "results/effect_size_audit/false_accept_forensics/accepted_false_accepts.csv",
    "results/effect_size_audit/false_accept_forensics/false_accept_tag_counts.csv",
    "reports/practical_utility_decision_gate.md",
    "docs/calibrated_protocol_lock_v1.md",
]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    _ensure_dir(target.parent)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_yaml(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return data


def _markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        if not df.empty:
            raise KeyError(f"missing columns for report table: {missing}")
        df = pd.DataFrame(columns=columns)
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


def load_cstr_weakness_config(path: str | Path) -> dict[str, Any]:
    config = _load_yaml(path)
    required = {
        "audit_id",
        "system_id",
        "source_artifacts",
        "primary_coverages",
        "granular_coverages",
        "bad_rmse_threshold",
        "practical_thresholds",
        "required_signals",
        "allowed_models",
        "allowed_judges",
        "diagnostic_oracle",
        "forbidden",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"missing CSTR weakness audit config keys: {missing}")
    if config["system_id"] != "cstr":
        raise ValueError("CSTR weakness audit is restricted to system_id='cstr'")
    if 0.01 not in [float(value) for value in config["granular_coverages"]]:
        raise ValueError("granular_coverages must include 0.01")
    if 0.02 not in [float(value) for value in config["granular_coverages"]]:
        raise ValueError("granular_coverages must include 0.02")
    practical = config["practical_thresholds"]
    if float(practical["minimum_absolute_far_reduction"]) != 0.05:
        raise ValueError("minimum_absolute_far_reduction must match previous audit: 0.05")
    if float(practical["minimum_relative_far_reduction"]) != 0.10:
        raise ValueError("minimum_relative_far_reduction must match previous audit: 0.10")
    forbidden = config["forbidden"]
    for key in [
        "allow_new_models",
        "allow_new_judges",
        "allow_new_signals",
        "allow_new_systems",
        "allow_protocol_mutation",
        "allow_overwrite_prior_artifacts",
    ]:
        if forbidden.get(key) is not False:
            raise ValueError(f"forbidden.{key} must be false")
    if set(config["required_signals"]) != set(DEPLOYABLE_SIGNALS):
        raise ValueError("required_signals must match existing deployable signal columns")
    unknown_judges = sorted(set(config["allowed_judges"]) - {BASELINE_JUDGE, *CALIBRATED_FAMILY, "support_only", "uncertainty_only", "disagreement_only", "invariant_only", "repair_only", "combined_linear", "random_baseline", "oracle_error_rank"})
    if unknown_judges:
        raise ValueError(f"unknown allowed judges: {unknown_judges}")
    unknown_models = sorted(set(config["allowed_models"]) - {"hold_last", "linear_narx", "mlp_state_space"})
    if unknown_models:
        raise ValueError(f"unknown allowed models: {unknown_models}")
    return config


def _accepted_mask(group: pd.DataFrame, risk_column: str, coverage: float) -> pd.Series:
    accepted_count = min(max(int(math.ceil(float(coverage) * len(group))), 1), len(group))
    accepted_index = group.sort_values(risk_column, kind="mergesort").head(accepted_count).index
    mask = pd.Series(False, index=group.index)
    mask.loc[accepted_index] = True
    return mask


def _auroc(labels: pd.Series | np.ndarray, scores: pd.Series | np.ndarray) -> float | None:
    y = np.asarray(labels, dtype=bool)
    s = np.asarray(scores, dtype=float)
    valid = np.isfinite(s)
    y = y[valid]
    s = s[valid]
    n_pos = int(np.sum(y))
    n_neg = int(np.sum(~y))
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = stats.rankdata(s, method="average")
    rank_sum_pos = float(np.sum(ranks[y]))
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _average_precision(labels: pd.Series | np.ndarray, scores: pd.Series | np.ndarray) -> float | None:
    y = np.asarray(labels, dtype=bool)
    s = np.asarray(scores, dtype=float)
    valid = np.isfinite(s)
    y = y[valid]
    s = s[valid]
    n_pos = int(np.sum(y))
    if n_pos == 0:
        return None
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(~y_sorted)
    precision = tp / np.maximum(tp + fp, 1)
    return float(np.sum(precision[y_sorted]) / n_pos)


def _overlap_coefficient(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> float | None:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return None
    if np.ptp(np.concatenate([x, y])) <= 1e-12:
        return 1.0
    bins = np.histogram_bin_edges(np.concatenate([x, y]), bins="auto")
    hx, _ = np.histogram(x, bins=bins, density=True)
    hy, _ = np.histogram(y, bins=bins, density=True)
    widths = np.diff(bins)
    return float(np.sum(np.minimum(hx, hy) * widths))


def _spearman(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float | None:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    if int(np.sum(valid)) < 2:
        return None
    if np.ptp(a[valid]) <= 1e-12 or np.ptp(b[valid]) <= 1e-12:
        return 0.0
    return float(stats.spearmanr(a[valid], b[valid]).correlation)


def _distribution(values: pd.Series | np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"mean": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0, "zero_fraction": 0.0}
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p10": float(np.quantile(x, 0.10)),
        "p90": float(np.quantile(x, 0.90)),
        "zero_fraction": float(np.mean(np.isclose(x, 0.0))),
    }


def _plot_bar(df: pd.DataFrame, x_col: str, y_col: str, output: Path, title: str, ylabel: str) -> None:
    _ensure_dir(output.parent)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    frame = df.sort_values(y_col, ascending=False)
    ax.bar(frame[x_col].astype(str), frame[y_col].astype(float))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, output: Path, title: str) -> None:
    _ensure_dir(output.parent)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(df[x_col].astype(float), df[y_col].astype(float), s=12, alpha=0.45)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _scan_forbidden_runtime_refs(paths: list[Path]) -> dict[str, list[str]]:
    old_repo_hits: list[str] = []
    path_hacks: list[str] = []
    for root in paths:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith(("import ", "from ")) and any(name.replace("-", "_") in stripped or name in stripped for name in OLD_REPO_NAMES):
                    old_repo_hits.append(str(path))
                    break
            for line in text.splitlines():
                stripped = line.strip()
                env_key = "PYTHON" + "PATH"
                if stripped.startswith("sys" + ".path") or stripped.startswith(f"os.environ[\"{env_key}\""):
                    path_hacks.append(str(path))
                    break
    return {"old_repo_runtime_import_hits": sorted(set(old_repo_hits)), "path_hack_hits": sorted(set(path_hacks))}


def verify_cstr_weakness_audit_preconditions(
    config_path: str | Path,
    output: str | Path,
    report_output: str | Path | None = "reports/cstr_weakness_precondition_check.md",
) -> dict[str, Any]:
    config = load_cstr_weakness_config(config_path)
    out_dir = Path(output)
    _ensure_dir(out_dir)
    missing = [path for path in PRIOR_ARTIFACTS if not Path(path).exists() or Path(path).stat().st_size == 0]
    gate_json_path = Path("reports/practical_utility_decision_gate.json")
    gate_json = _load_json(gate_json_path) if gate_json_path.exists() else {}
    decision = gate_json.get("decision", "")
    expansion_allowed = bool(gate_json.get("expansion_allowed", True))
    dirty_lines = []
    git_dir = Path(".git")
    if git_dir.exists():
        import subprocess

        status = subprocess.run(["git", "status", "--short"], check=True, capture_output=True, text=True)
        dirty_lines = [line for line in status.stdout.splitlines() if line.strip()]
    scan = _scan_forbidden_runtime_refs([Path("src"), Path("scripts")])
    forbidden_evidence_refs = []
    audit_text = yaml.safe_dump(config).lower()
    if "heat_exchanger" in audit_text or "rssm" in audit_text:
        forbidden_evidence_refs.append("config references forbidden evidence")
    verdict = "READY_FOR_CSTR_WEAKNESS_AUDIT"
    reasons = []
    if missing:
        verdict = "NOT_READY"
        reasons.append(f"missing artifacts: {missing}")
    if decision != "NARROW_TO_WEAK_LOW_COVERAGE_CLAIM":
        verdict = "NOT_READY"
        reasons.append(f"practical gate decision is {decision!r}")
    if expansion_allowed:
        verdict = "NOT_READY"
        reasons.append("expansion_allowed is not false")
    if scan["old_repo_runtime_import_hits"] or scan["path_hack_hits"]:
        verdict = "NOT_READY"
        reasons.append("forbidden runtime dependency/path scan failed")
    if forbidden_evidence_refs:
        verdict = "NOT_READY"
        reasons.extend(forbidden_evidence_refs)
    result = {
        "audit_id": config["audit_id"],
        "working_tree_dirty": bool(dirty_lines),
        "dirty_state": dirty_lines,
        "current_controlling_decision": decision,
        "expansion_allowed": expansion_allowed,
        "required_artifacts": [{"path": path, "exists": path not in missing} for path in PRIOR_ARTIFACTS],
        "protocol_lock_exists": Path("docs/calibrated_protocol_lock_v1.md").exists(),
        "forbidden_dependency_scan": scan,
        "artifact_mutation_policy": "prior evidence directories are read-only for this audit",
        "verdict": verdict,
        "reasons": reasons,
    }
    _write_json(out_dir / "precondition_check.json", result)
    if report_output is not None:
        write_precondition_report(result, Path(report_output))
    if verdict != "READY_FOR_CSTR_WEAKNESS_AUDIT":
        raise RuntimeError(f"CSTR weakness audit preconditions failed: {reasons}")
    return result


def write_precondition_report(result: dict[str, Any], output: Path) -> None:
    artifacts = pd.DataFrame(result["required_artifacts"])
    scan = result["forbidden_dependency_scan"]
    text = f"""# CSTR Weakness Audit Preconditions

## Current controlling decision

{result["current_controlling_decision"]}

## Expansion status

Expansion allowed: {result["expansion_allowed"]}

## Required artifacts

{_markdown_table(artifacts, ["path", "exists"])}

## Protocol lock status

Protocol lock exists: {result["protocol_lock_exists"]}

## Forbidden dependency scan

Old repo runtime import hits: {scan["old_repo_runtime_import_hits"] or "none"}
Path hack hits: {scan["path_hack_hits"] or "none"}

## Artifact mutation policy

{result["artifact_mutation_policy"]}

## Verdict

{result["verdict"]}
"""
    _ensure_dir(output.parent)
    output.write_text(text, encoding="utf-8")


def _cstr_statewise_table(config: dict[str, Any]) -> pd.DataFrame:
    experiment_config = _load_yaml("configs/experiments/calibrated_cstr.yaml")
    dataset = load_dataset("results/calibrated_cstr/data")
    models = [make_model(str(model_id), seed=int(experiment_config["seed"]) + idx) for idx, model_id in enumerate(experiment_config["models"])]
    for model in models:
        model.fit(dataset["model_train"])
    rows = []
    for split, batch in dataset.items():
        if not split.startswith("judge_test"):
            continue
        for i in range(batch.n_trajectories):
            scenario_id = f"{split}_{i:04d}"
            actual = batch.states[i]
            for model in models:
                predicted = model.predict_rollout(batch.states[i, 0], batch.actions[i], batch.disturbances[i])
                error = predicted - actual
                rows.append(
                    {
                        "scenario_id": scenario_id,
                        "model_id": model.model_id,
                        "state_error_concentration_rmse": float(np.sqrt(np.mean(error[:, 0] ** 2))),
                        "state_error_temperature_rmse": float(np.sqrt(np.mean(error[:, 1] ** 2))),
                        "state_error_concentration_max_abs": float(np.max(np.abs(error[:, 0]))),
                        "state_error_temperature_max_abs": float(np.max(np.abs(error[:, 1]))),
                        "state_error_concentration_final": float(abs(error[-1, 0])),
                        "state_error_temperature_final": float(abs(error[-1, 1])),
                    }
                )
    return pd.DataFrame(rows)


def _event_label_cstr_table() -> pd.DataFrame:
    path = Path("results/effect_size_audit/event_risk/event_labels.csv")
    if not path.exists():
        return pd.DataFrame()
    labels = pd.read_csv(path)
    labels = labels[labels["system_id"] == "cstr"].copy()
    event_cols = [
        col
        for col in labels.columns
        if col in {"scenario_id", "model_id", "true_any_event", "predicted_any_event", "bad_event"}
        or col.startswith(("true_temperature_", "predicted_temperature_", "bad_temperature_"))
        or col.startswith(("true_concentration_", "predicted_concentration_", "bad_concentration_"))
        or col.startswith(("true_unsafe_", "predicted_unsafe_", "bad_unsafe_"))
    ]
    return labels[event_cols].copy()


def _selected_calibrated_for_group(table: pd.DataFrame, coverage: float, family: list[str]) -> pd.DataFrame:
    rows = []
    for (model_id, scenario_type), group in table.groupby(["model_id", "scenario_type"], sort=False):
        best_judge = None
        best_far = None
        for judge_id in family:
            risk_col = f"risk_{judge_id}"
            accepted = _accepted_mask(group, risk_col, coverage)
            accepted_rows = group.loc[accepted]
            far = float(accepted_rows["bad_rmse_label"].astype(bool).mean())
            if best_far is None or far < best_far - 1e-12 or (abs(far - best_far) <= 1e-12 and judge_id < str(best_judge)):
                best_far = far
                best_judge = judge_id
        rows.append({"model_id": model_id, "scenario_type": scenario_type, "coverage": float(coverage), "calibrated_judge_id": best_judge})
    return pd.DataFrame(rows)


def build_cstr_diagnosis_table(config_path: str | Path, output: str | Path) -> dict[str, Any]:
    config = load_cstr_weakness_config(config_path)
    out_dir = Path(output)
    _ensure_dir(out_dir)
    test = pd.read_csv("results/calibrated_cstr/test_table.csv")
    test = test[test["system_id"] == "cstr"].copy()
    if test.empty:
        raise RuntimeError("calibrated CSTR test table is empty")
    allowed_models = set(config["allowed_models"])
    test = test[test["model_id"].isin(allowed_models)].copy()
    if set(test["model_id"].unique()) != allowed_models:
        raise RuntimeError("not all allowed CSTR models are present")
    threshold = float(config["bad_rmse_threshold"])
    test["bad_rmse_label"] = test["rmse"].astype(float) > threshold
    statewise = _cstr_statewise_table(config)
    trajectory_available = not statewise.empty and all(column in statewise.columns for column in STATEWISE_COLUMNS)
    if trajectory_available:
        test = test.merge(statewise, on=["scenario_id", "model_id"], how="left")
    else:
        for column in STATEWISE_COLUMNS:
            test[column] = np.nan
    event_labels = _event_label_cstr_table()
    event_labels_available = not event_labels.empty
    if event_labels_available:
        test = test.merge(event_labels, on=["scenario_id", "model_id"], how="left")
    coverages = [float(value) for value in config["granular_coverages"]]
    allowed_judges = [str(value) for value in config["allowed_judges"]]
    family = [judge for judge in CALIBRATED_FAMILY if judge in allowed_judges]
    forensic = pd.read_csv("results/effect_size_audit/false_accept_forensics/accepted_false_accepts.csv")
    forensic = forensic[forensic["system_id"] == "cstr"][["scenario_id", "model_id", "coverage", "tags"]].copy()
    selected = pd.concat([_selected_calibrated_for_group(test, coverage, family) for coverage in coverages], ignore_index=True)
    frames = []
    for coverage in coverages:
        selected_cov = selected[np.isclose(selected["coverage"], coverage)]
        for judge_id in allowed_judges:
            risk_col = f"risk_{judge_id}"
            if risk_col not in test.columns:
                raise ValueError(f"missing risk column for allowed judge {judge_id}")
            chunk = test.copy()
            chunk["judge_id"] = judge_id
            chunk["coverage"] = float(coverage)
            chunk["risk_score"] = chunk[risk_col].astype(float)
            accepted = []
            for _, group in chunk.groupby(["model_id", "scenario_type"], sort=False):
                accepted.append(_accepted_mask(group, "risk_score", coverage))
            accepted_mask = pd.concat(accepted).sort_index()
            chunk["accepted"] = accepted_mask.reindex(chunk.index).astype(bool)
            chunk["false_accept"] = chunk["accepted"] & chunk["bad_rmse_label"].astype(bool)
            chunk = chunk.merge(selected_cov, on=["model_id", "scenario_type", "coverage"], how="left")
            chunk["baseline_judge_id"] = BASELINE_JUDGE
            frames.append(chunk)
    table = pd.concat(frames, ignore_index=True)
    table = table.merge(forensic, on=["scenario_id", "model_id", "coverage"], how="left")
    table["forensic_tags"] = table["tags"].fillna("none").replace("", "none")
    table = table.drop(columns=["tags"])
    table["trajectory_available"] = bool(trajectory_available)
    table["event_labels_available"] = bool(event_labels_available)
    required = [
        "system_id",
        "scenario_id",
        "scenario_type",
        "model_id",
        "judge_id",
        "coverage",
        "accepted",
        "false_accept",
        "bad_rmse_label",
        "rmse",
        "mae",
        "max_abs_error",
        "final_state_error",
        "support_distance",
        "uncertainty_score",
        "disagreement_score",
        "invariant_residual",
        "repair_amount",
        "risk_score",
        "calibrated_judge_id",
        "baseline_judge_id",
        "forensic_tags",
        "trajectory_available",
        "event_labels_available",
        *STATEWISE_COLUMNS,
    ]
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise RuntimeError(f"diagnosis table missing required columns: {missing}")
    table.to_csv(out_dir / "cstr_diagnosis_table.csv", index=False)
    schema = {
        "row_count": int(len(table)),
        "scenario_count": int(test["scenario_id"].nunique()),
        "models": sorted(test["model_id"].unique().tolist()),
        "judges": allowed_judges,
        "coverages": coverages,
        "scenario_types": sorted(test["scenario_type"].unique().tolist()),
        "trajectory_available": bool(trajectory_available),
        "event_labels_available": bool(event_labels_available),
        "required_columns": required,
        "missing_columns": missing,
        "verdict": "ACCEPTED" if not missing and len(table) > 0 else "REJECTED",
    }
    _write_json(out_dir / "cstr_diagnosis_schema.json", schema)
    write_diagnosis_table_report(schema, Path("reports/cstr_diagnosis_table_report.md"))
    return schema


def write_diagnosis_table_report(schema: dict[str, Any], output: Path) -> None:
    text = f"""# CSTR Diagnosis Table Report

## Input artifacts

results/calibrated_cstr/test_table.csv; results/effect_size_audit/false_accept_forensics/accepted_false_accepts.csv; results/effect_size_audit/event_risk/event_labels.csv

## Output table

results/cstr_weakness_audit/diagnosis_table/cstr_diagnosis_table.csv

## Row count

{schema["row_count"]}

## Scenario count

{schema["scenario_count"]}

## Models

{", ".join(schema["models"])}

## Judges

{", ".join(schema["judges"])}

## Coverages

{", ".join(map(str, schema["coverages"]))}

## Scenario types

{", ".join(schema["scenario_types"])}

## Trajectory availability

{schema["trajectory_available"]}

## Event-label availability

{schema["event_labels_available"]}

## Missing columns

{schema["missing_columns"] or "none"}

## Verdict

{schema["verdict"]}
"""
    _ensure_dir(output.parent)
    output.write_text(text, encoding="utf-8")


def _selected_calibrated_rows(table: pd.DataFrame, coverages: list[float] | None = None) -> pd.DataFrame:
    frame = table[table["judge_id"] == table["calibrated_judge_id"]].copy()
    if coverages is not None:
        frame = frame[frame["coverage"].isin([float(value) for value in coverages])].copy()
    return frame


def _base_rows(table: pd.DataFrame) -> pd.DataFrame:
    return table.drop_duplicates(["scenario_id", "model_id"]).copy()


def _dominant_state(conc: float, temp: float) -> tuple[str, float, float, float]:
    conc_norm = float(conc) / 2.0
    temp_norm = float(temp) / 250.0
    if conc_norm >= temp_norm:
        ratio = conc_norm / max(temp_norm, 1e-12)
        return "concentration", ratio, conc_norm, temp_norm
    ratio = temp_norm / max(conc_norm, 1e-12)
    return "temperature", ratio, conc_norm, temp_norm


def _state_summary(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[*group_cols, "concentration_rmse_mean", "temperature_rmse_mean", "dominant_state", "dominant_state_ratio"])
    grouped = frame.groupby(group_cols, as_index=False).agg(
        concentration_rmse_mean=("state_error_concentration_rmse", "mean"),
        temperature_rmse_mean=("state_error_temperature_rmse", "mean"),
        concentration_rmse_median=("state_error_concentration_rmse", "median"),
        temperature_rmse_median=("state_error_temperature_rmse", "median"),
        concentration_max_abs_mean=("state_error_concentration_max_abs", "mean"),
        temperature_max_abs_mean=("state_error_temperature_max_abs", "mean"),
        concentration_final_error_mean=("state_error_concentration_final", "mean"),
        temperature_final_error_mean=("state_error_temperature_final", "mean"),
    )
    states = grouped.apply(lambda row: _dominant_state(row["concentration_rmse_mean"], row["temperature_rmse_mean"]), axis=1)
    grouped["dominant_state"] = [item[0] for item in states]
    grouped["dominant_state_ratio"] = [item[1] for item in states]
    return grouped


def _statewise_verdict(selected: pd.DataFrame) -> str:
    false_accepts = selected[selected["false_accept"]].copy()
    if false_accepts.empty:
        return "TRAJECTORY_DATA_UNAVAILABLE"
    conc_norm = false_accepts["state_error_concentration_rmse"].mean() / 2.0
    temp_norm = false_accepts["state_error_temperature_rmse"].mean() / 250.0
    total = conc_norm + temp_norm
    if total <= 1e-12:
        return "BOTH_STATES"
    if temp_norm / total >= 0.65:
        return "TEMPERATURE_DOMINATED"
    if conc_norm / total >= 0.65:
        return "CONCENTRATION_DOMINATED"
    return "BOTH_STATES"


def analyze_cstr_statewise_error(diagnosis_table: str | Path, config_path: str | Path, output: str | Path) -> dict[str, Any]:
    _ = load_cstr_weakness_config(config_path)
    table = pd.read_csv(diagnosis_table, low_memory=False)
    out_dir = Path(output)
    _ensure_dir(out_dir)
    if not set(STATEWISE_COLUMNS).issubset(table.columns) or not bool(table.get("trajectory_available", pd.Series([False])).all()):
        summary = {"verdict": "TRAJECTORY_DATA_UNAVAILABLE", "key_finding": "state-wise trajectory diagnostics are unavailable"}
        _write_json(out_dir / "statewise_error_summary.json", summary)
        pd.DataFrame().to_csv(out_dir / "statewise_error_summary.csv", index=False)
        pd.DataFrame().to_csv(out_dir / "statewise_false_accepts.csv", index=False)
        write_statewise_report(summary, pd.DataFrame(), pd.DataFrame(), Path("reports/cstr_statewise_error.md"))
        return summary
    selected = _selected_calibrated_rows(table, [0.05, 0.10])
    summary_by_model = _state_summary(selected, ["model_id"])
    summary_by_scenario = _state_summary(selected, ["scenario_type"])
    by_all = _state_summary(selected, ["model_id", "scenario_type", "coverage", "accepted", "false_accept"])
    by_all.to_csv(out_dir / "statewise_error_summary.csv", index=False)
    false_accepts = selected[selected["false_accept"]].copy()
    fa_summary = _state_summary(false_accepts, ["coverage", "model_id", "scenario_type"])
    fa_summary.to_csv(out_dir / "statewise_false_accepts.csv", index=False)
    verdict = _statewise_verdict(selected)
    false_accepts = false_accepts.copy()
    if not false_accepts.empty:
        conc_norm = float(false_accepts["state_error_concentration_rmse"].mean() / 2.0)
        temp_norm = float(false_accepts["state_error_temperature_rmse"].mean() / 250.0)
        total = max(conc_norm + temp_norm, 1e-12)
    else:
        conc_norm = temp_norm = total = 0.0
    summary = {
        "verdict": verdict,
        "key_finding": f"accepted false-accept normalized error share concentration={conc_norm / total if total else 0.0:.6f}, temperature={temp_norm / total if total else 0.0:.6f}",
        "accepted_false_accept_count": int(len(false_accepts)),
        "accepted_false_accept_concentration_share": float(conc_norm / total) if total else 0.0,
        "accepted_false_accept_temperature_share": float(temp_norm / total) if total else 0.0,
    }
    _write_json(out_dir / "statewise_error_summary.json", summary)
    _plot_bar(summary_by_model, "model_id", "temperature_rmse_mean", out_dir / "statewise_error_by_model.png", "CSTR temperature RMSE by model", "temperature_rmse")
    _plot_bar(summary_by_scenario, "scenario_type", "temperature_rmse_mean", out_dir / "statewise_error_by_scenario.png", "CSTR temperature RMSE by scenario", "temperature_rmse")
    write_statewise_report(summary, summary_by_model, summary_by_scenario, Path("reports/cstr_statewise_error.md"), fa_summary)
    return summary


def write_statewise_report(
    summary: dict[str, Any],
    by_model: pd.DataFrame,
    by_scenario: pd.DataFrame,
    output: Path,
    false_accepts: pd.DataFrame | None = None,
) -> None:
    false_accepts = false_accepts if false_accepts is not None else pd.DataFrame()
    text = f"""# CSTR State-Wise Error Decomposition

## Question

Is CSTR weakness dominated by concentration error, temperature error, or both?

## State-wise error by model

{_markdown_table(by_model.rename(columns={"model_id": "model", "concentration_rmse_mean": "concentration_rmse", "temperature_rmse_mean": "temperature_rmse"}), ["model", "concentration_rmse", "temperature_rmse", "dominant_state"])}

## State-wise error by scenario type

{_markdown_table(by_scenario.rename(columns={"concentration_rmse_mean": "concentration_rmse", "temperature_rmse_mean": "temperature_rmse"}), ["scenario_type", "concentration_rmse", "temperature_rmse", "dominant_state"], max_rows=40)}

## Accepted false accepts

{_markdown_table(false_accepts.rename(columns={"model_id": "model", "concentration_rmse_mean": "concentration_rmse", "temperature_rmse_mean": "temperature_rmse"}), ["coverage", "model", "scenario_type", "concentration_rmse", "temperature_rmse", "dominant_state"], max_rows=40)}

## Verdict

{summary["verdict"]}
"""
    _ensure_dir(output.parent)
    output.write_text(text, encoding="utf-8")


def _selected_primary(table: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    return _selected_calibrated_rows(table, [float(value) for value in config["primary_coverages"]])


def _repair_metrics_verdict(metrics: dict[str, Any]) -> str:
    auroc = metrics.get("repair_auroc_for_bad_rmse_label")
    low_frac = float(metrics.get("fraction_accepted_false_accepts_with_low_repair", 0.0))
    dynamic_range = float(metrics.get("repair_dynamic_range", 0.0))
    out_of_bounds = bool(metrics.get("out_of_bounds_or_repair_events_observed", False))
    if dynamic_range <= 1e-12 and out_of_bounds:
        return "REPAIR_SIGNAL_MISCOMPUTED"
    if auroc is not None and float(auroc) < 0.60 and low_frac >= 0.50:
        return "REPAIR_SIGNAL_BLIND_SPOT"
    if auroc is not None and float(auroc) >= 0.70 and low_frac <= 0.10:
        return "REPAIR_SIGNAL_USEFUL"
    corr = abs(float(metrics.get("repair_spearman_with_rmse") or 0.0))
    if dynamic_range <= 1e-12 or corr < 0.20:
        return "REPAIR_SIGNAL_IRRELEVANT_TO_RMSE"
    return "INCONCLUSIVE"


def analyze_cstr_repair_signal(diagnosis_table: str | Path, config_path: str | Path, output: str | Path) -> dict[str, Any]:
    config = load_cstr_weakness_config(config_path)
    table = pd.read_csv(diagnosis_table, low_memory=False)
    selected = _selected_primary(table, config)
    base = _base_rows(table)
    out_dir = Path(output)
    _ensure_dir(out_dir)
    groups = {
        "accepted_good": selected[selected["accepted"] & ~selected["bad_rmse_label"].astype(bool)]["repair_amount"],
        "accepted_bad": selected[selected["accepted"] & selected["bad_rmse_label"].astype(bool)]["repair_amount"],
        "rejected_good": selected[~selected["accepted"] & ~selected["bad_rmse_label"].astype(bool)]["repair_amount"],
        "rejected_bad": selected[~selected["accepted"] & selected["bad_rmse_label"].astype(bool)]["repair_amount"],
    }
    distribution = pd.DataFrame([{"group": name, **_distribution(values)} for name, values in groups.items()])
    distribution.to_csv(out_dir / "repair_signal_distribution.csv", index=False)
    repair = base["repair_amount"].astype(float)
    bad = base["bad_rmse_label"].astype(bool)
    low_threshold = float(np.quantile(repair, 0.25)) if len(repair) else 0.0
    false_accepts = selected[selected["false_accept"]].copy()
    severe = false_accepts[false_accepts["rmse"].astype(float) >= float(config["bad_rmse_threshold"]) * 3.0]
    low_repair_false_accepts = false_accepts[false_accepts["repair_amount"].astype(float) <= low_threshold + 1e-12]
    severe_low = severe[severe["repair_amount"].astype(float) <= low_threshold + 1e-12]
    low_by_cov = []
    for coverage, frame in false_accepts.groupby("coverage", sort=True):
        low_count = int((frame["repair_amount"].astype(float) <= low_threshold + 1e-12).sum())
        low_by_cov.append(
            {
                "coverage": float(coverage),
                "accepted_false_accept_count": int(len(frame)),
                "low_repair_false_accept_count": low_count,
                "fraction": float(low_count / len(frame)) if len(frame) else 0.0,
            }
        )
    repair_only = table[(table["judge_id"] == "repair_only") & table["coverage"].isin([float(v) for v in config["primary_coverages"]])]
    calibrated = selected.copy()
    repair_comp = []
    for coverage in [float(v) for v in config["primary_coverages"]]:
        r = repair_only[np.isclose(repair_only["coverage"], coverage)]
        c = calibrated[np.isclose(calibrated["coverage"], coverage)]
        repair_comp.append(
            {
                "coverage": coverage,
                "repair_only_far": float(r[r["accepted"]]["bad_rmse_label"].astype(bool).mean()) if len(r[r["accepted"]]) else 0.0,
                "calibrated_far": float(c[c["accepted"]]["bad_rmse_label"].astype(bool).mean()) if len(c[c["accepted"]]) else 0.0,
            }
        )
    metrics = {
        "repair_auroc_for_bad_rmse_label": _auroc(bad, repair),
        "repair_auprc_for_bad_rmse_label": _average_precision(bad, repair),
        "repair_spearman_with_rmse": _spearman(repair, base["rmse"]),
        "repair_low_risk_threshold": low_threshold,
        "repair_false_negative_rate_at_low_risk_threshold": float(((bad) & (repair <= low_threshold + 1e-12)).sum() / max(int(bad.sum()), 1)),
        "fraction_accepted_false_accepts_with_low_repair": float(len(low_repair_false_accepts) / len(false_accepts)) if len(false_accepts) else 0.0,
        "fraction_severe_false_accepts_with_low_repair": float(len(severe_low) / len(severe)) if len(severe) else 0.0,
        "zero_repair_fraction": float(np.isclose(repair, 0.0).mean()) if len(repair) else 0.0,
        "near_zero_repair_fraction": float((repair <= 1e-8).mean()) if len(repair) else 0.0,
        "repair_dynamic_range": float(repair.max() - repair.min()) if len(repair) else 0.0,
        "out_of_bounds_or_repair_events_observed": bool(float(repair.max()) > 1e-12) if len(repair) else False,
        "low_repair_false_accepts_by_coverage": low_by_cov,
        "repair_only_judge_comparison": repair_comp,
    }
    metrics["verdict"] = _repair_metrics_verdict(metrics)
    metrics["key_finding"] = f"repair_amount AUROC={metrics['repair_auroc_for_bad_rmse_label']}, low-repair accepted false-accept fraction={metrics['fraction_accepted_false_accepts_with_low_repair']:.6f}"
    _write_json(out_dir / "repair_signal_metrics.json", metrics)
    _plot_repair_overlap(selected, out_dir / "repair_signal_overlap.png")
    _plot_scatter(base, "repair_amount", "rmse", out_dir / "repair_vs_rmse.png", "CSTR repair_amount vs RMSE")
    write_repair_report(metrics, distribution, pd.DataFrame(low_by_cov), pd.DataFrame(repair_comp), Path("reports/cstr_repair_signal_audit.md"))
    return metrics


def _plot_repair_overlap(selected: pd.DataFrame, output: Path) -> None:
    _ensure_dir(output.parent)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    good = selected[selected["accepted"] & ~selected["bad_rmse_label"].astype(bool)]["repair_amount"].astype(float)
    bad = selected[selected["accepted"] & selected["bad_rmse_label"].astype(bool)]["repair_amount"].astype(float)
    ax.hist(good, bins=20, alpha=0.6, label="accepted_good")
    ax.hist(bad, bins=20, alpha=0.6, label="accepted_bad")
    ax.set_xlabel("repair_amount")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def write_repair_report(metrics: dict[str, Any], distribution: pd.DataFrame, low_by_cov: pd.DataFrame, repair_comp: pd.DataFrame, output: Path) -> None:
    metric_table = pd.DataFrame(
        [
            {"metric": key, "value": value}
            for key, value in metrics.items()
            if key
            in {
                "repair_auroc_for_bad_rmse_label",
                "repair_auprc_for_bad_rmse_label",
                "repair_spearman_with_rmse",
                "repair_false_negative_rate_at_low_risk_threshold",
                "fraction_accepted_false_accepts_with_low_repair",
                "fraction_severe_false_accepts_with_low_repair",
                "zero_repair_fraction",
                "near_zero_repair_fraction",
                "repair_dynamic_range",
            }
        ]
    )
    text = f"""# CSTR Repair-Signal Blind-Spot Audit

## Question

Does repair_amount protect against CSTR false accepts?

## Repair distribution

{_markdown_table(distribution, ["group", "mean", "median", "p10", "p90", "zero_fraction"])}

## Repair as failure detector

{_markdown_table(metric_table, ["metric", "value"])}

## Low-repair false accepts

{_markdown_table(low_by_cov, ["coverage", "accepted_false_accept_count", "low_repair_false_accept_count", "fraction"])}

## Repair-only judge comparison

{_markdown_table(repair_comp, ["coverage", "repair_only_far", "calibrated_far"])}

## Interpretation

{metrics["key_finding"]}

## Verdict

{metrics["verdict"]}
"""
    _ensure_dir(output.parent)
    output.write_text(text, encoding="utf-8")


def _signal_group(table: pd.DataFrame) -> pd.Series:
    return np.select(
        [
            table["accepted"].astype(bool) & ~table["bad_rmse_label"].astype(bool),
            table["accepted"].astype(bool) & table["bad_rmse_label"].astype(bool),
            ~table["accepted"].astype(bool) & ~table["bad_rmse_label"].astype(bool),
            ~table["accepted"].astype(bool) & table["bad_rmse_label"].astype(bool),
        ],
        ["accepted_good", "accepted_bad", "rejected_good", "rejected_bad"],
        default="unknown",
    )


def _cohens_d(a: pd.Series, b: pd.Series) -> float | None:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if len(x) < 2 or len(y) < 2:
        return None
    pooled = math.sqrt(((len(x) - 1) * np.var(x, ddof=1) + (len(y) - 1) * np.var(y, ddof=1)) / max(len(x) + len(y) - 2, 1))
    if pooled <= 1e-12:
        return 0.0
    return float((np.mean(y) - np.mean(x)) / pooled)


def _signal_metrics_for_frame(frame: pd.DataFrame, signal: str) -> dict[str, Any]:
    accepted_good = frame[frame["signal_group"] == "accepted_good"][signal]
    accepted_bad = frame[frame["signal_group"] == "accepted_bad"][signal]
    too_few = len(accepted_good) < 2 or len(accepted_bad) < 2
    if too_few:
        return {"signal": signal, "auroc": None, "cohens_d": None, "overlap_coefficient": None, "mann_whitney_u_pvalue": None, "verdict": "TOO_FEW_SAMPLES"}
    labels = pd.concat([pd.Series(False, index=accepted_good.index), pd.Series(True, index=accepted_bad.index)])
    scores = pd.concat([accepted_good, accepted_bad])
    auroc = _auroc(labels, scores)
    d = _cohens_d(accepted_good, accepted_bad)
    overlap = _overlap_coefficient(accepted_good, accepted_bad)
    try:
        pvalue = float(stats.mannwhitneyu(accepted_good, accepted_bad, alternative="two-sided").pvalue)
    except Exception:
        pvalue = None
    verdict = "SEPARATES" if auroc is not None and auroc >= 0.75 else "MIXED" if auroc is not None and auroc >= 0.60 else "FAILS"
    return {"signal": signal, "auroc": auroc, "cohens_d": d, "overlap_coefficient": overlap, "mann_whitney_u_pvalue": pvalue, "verdict": verdict}


def analyze_cstr_signal_overlap(diagnosis_table: str | Path, config_path: str | Path, output: str | Path) -> dict[str, Any]:
    config = load_cstr_weakness_config(config_path)
    table = pd.read_csv(diagnosis_table, low_memory=False)
    selected = _selected_primary(table, config).copy()
    selected["signal_group"] = _signal_group(selected)
    signals = [*DEPLOYABLE_SIGNALS, "risk_score"]
    out_dir = Path(output)
    _ensure_dir(out_dir)
    summary_rows = []
    for signal in signals:
        for group, frame in selected.groupby("signal_group", sort=True):
            values = frame[signal].astype(float)
            row = {"signal": signal, "group": group}
            row.update(
                {
                    "mean": float(values.mean()) if len(values) else 0.0,
                    "median": float(values.median()) if len(values) else 0.0,
                    "p10": float(values.quantile(0.10)) if len(values) else 0.0,
                    "p25": float(values.quantile(0.25)) if len(values) else 0.0,
                    "p75": float(values.quantile(0.75)) if len(values) else 0.0,
                    "p90": float(values.quantile(0.90)) if len(values) else 0.0,
                    "count": int(len(values)),
                }
            )
            summary_rows.append(row)
    overlap_summary = pd.DataFrame(summary_rows)
    overlap_summary.to_csv(out_dir / "signal_overlap_summary.csv", index=False)
    metrics = [_signal_metrics_for_frame(selected, signal) for signal in signals]
    deployable_metrics = [item for item in metrics if item["signal"] in DEPLOYABLE_SIGNALS and item["auroc"] is not None]
    if not deployable_metrics:
        verdict = "INCONCLUSIVE"
        best = None
    else:
        best = max(deployable_metrics, key=lambda item: float(item["auroc"]))
        if float(best["auroc"]) >= 0.75:
            verdict = "SIGNALS_SEPARATE_ACCEPTED_FAILURES"
        elif float(best["auroc"]) >= 0.60:
            verdict = "MIXED_SEPARABILITY"
        else:
            verdict = "SIGNAL_BLIND_SPOT"
    summary = {
        "verdict": verdict,
        "best_deployable_signal": None if best is None else best["signal"],
        "best_deployable_signal_auroc": None if best is None else best["auroc"],
        "accepted_good_count": int((selected["signal_group"] == "accepted_good").sum()),
        "accepted_bad_count": int((selected["signal_group"] == "accepted_bad").sum()),
        "metrics": metrics,
        "key_finding": "too few accepted samples" if best is None else f"best deployable accepted-region signal is {best['signal']} with AUROC {best['auroc']:.6f}",
    }
    _write_json(out_dir / "signal_separability_metrics.json", summary)
    _plot_signal_boxplots(selected, signals, out_dir / "signal_overlap_boxplots.png")
    write_signal_overlap_report(summary, pd.DataFrame(metrics), Path("reports/cstr_signal_overlap.md"))
    return summary


def _plot_signal_boxplots(selected: pd.DataFrame, signals: list[str], output: Path) -> None:
    _ensure_dir(output.parent)
    fig, axes = plt.subplots(len(signals), 1, figsize=(8, max(4, len(signals) * 2.0)))
    if len(signals) == 1:
        axes = [axes]
    for ax, signal in zip(axes, signals, strict=True):
        data = [
            selected[selected["signal_group"] == "accepted_good"][signal].astype(float),
            selected[selected["signal_group"] == "accepted_bad"][signal].astype(float),
        ]
        try:
            ax.boxplot(data, tick_labels=["accepted_good", "accepted_bad"], showfliers=False)
        except TypeError:  # matplotlib<3.9 compatibility.
            ax.boxplot(data, labels=["accepted_good", "accepted_bad"], showfliers=False)
        ax.set_title(signal)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def write_signal_overlap_report(summary: dict[str, Any], metrics: pd.DataFrame, output: Path) -> None:
    text = f"""# CSTR Signal Overlap and Separability Audit

## Question

Can current signals separate accepted-good from accepted-bad CSTR scenarios?

## Accepted-good vs accepted-bad separability

{_markdown_table(metrics, ["signal", "auroc", "cohens_d", "overlap_coefficient", "verdict"])}

## Signals that separate

{", ".join(metrics[metrics["verdict"] == "SEPARATES"]["signal"].astype(str).tolist()) or "none"}

## Signals that fail

{", ".join(metrics[metrics["verdict"] == "FAILS"]["signal"].astype(str).tolist()) or "none"}

## Interpretation

{summary["key_finding"]}

## Verdict

{summary["verdict"]}
"""
    _ensure_dir(output.parent)
    output.write_text(text, encoding="utf-8")


def _failure_summary(frame: pd.DataFrame, group_col: str, threshold: float) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[group_col, "accepted_count", "accepted_false_accept_count", "accepted_false_accept_rate", "mean_rmse", "mean_accepted_rmse", "mean_accepted_bad_rmse", "severe_misclassification_count"])
    rows = []
    for value, group in frame.groupby(group_col, sort=True):
        accepted = group[group["accepted"]]
        accepted_bad = accepted[accepted["false_accept"]]
        rows.append(
            {
                group_col: value,
                "accepted_count": int(len(accepted)),
                "accepted_false_accept_count": int(len(accepted_bad)),
                "accepted_false_accept_rate": float(len(accepted_bad) / len(accepted)) if len(accepted) else 0.0,
                "mean_rmse": float(group["rmse"].mean()) if len(group) else 0.0,
                "mean_accepted_rmse": float(accepted["rmse"].mean()) if len(accepted) else 0.0,
                "mean_accepted_bad_rmse": float(accepted_bad["rmse"].mean()) if len(accepted_bad) else 0.0,
                "severe_misclassification_count": int((accepted_bad["rmse"].astype(float) >= threshold * 3.0).sum()),
            }
        )
    return pd.DataFrame(rows)


def _model_scenario_verdict(summary: dict[str, Any]) -> str:
    total = int(summary["accepted_false_accept_count"])
    if total < 5:
        return "INCONCLUSIVE"
    if float(summary["top_model_share_of_false_accepts"]) >= 0.60:
        return "MODEL_SPECIFIC_CSTR_FAILURE"
    if float(summary["top_scenario_share_of_false_accepts"]) >= 0.60:
        return "SCENARIO_SPECIFIC_CSTR_FAILURE"
    return "DIFFUSE_CSTR_FAILURE"


def analyze_cstr_model_scenario_failures(diagnosis_table: str | Path, config_path: str | Path, output: str | Path) -> dict[str, Any]:
    config = load_cstr_weakness_config(config_path)
    table = pd.read_csv(diagnosis_table, low_memory=False)
    selected = _selected_primary(table, config)
    out_dir = Path(output)
    _ensure_dir(out_dir)
    threshold = float(config["bad_rmse_threshold"])
    model_summary = _failure_summary(selected, "model_id", threshold).rename(columns={"model_id": "model"})
    scenario_summary = _failure_summary(selected, "scenario_type", threshold)
    model_summary.to_csv(out_dir / "model_failure_summary.csv", index=False)
    scenario_summary.to_csv(out_dir / "scenario_failure_summary.csv", index=False)
    false_accepts = selected[selected["false_accept"]]
    total = int(len(false_accepts))
    model_counts = false_accepts["model_id"].value_counts()
    scenario_counts = false_accepts["scenario_type"].value_counts()
    dominant_model = str(model_counts.index[0]) if total else "none"
    dominant_scenario = str(scenario_counts.index[0]) if total else "none"
    summary = {
        "accepted_false_accept_count": total,
        "dominant_failed_model": dominant_model,
        "dominant_failed_scenario_type": dominant_scenario,
        "top_model_share_of_false_accepts": float(model_counts.iloc[0] / total) if total else 0.0,
        "top_scenario_share_of_false_accepts": float(scenario_counts.iloc[0] / total) if total else 0.0,
    }
    summary["verdict"] = _model_scenario_verdict(summary)
    summary["key_finding"] = f"top model share={summary['top_model_share_of_false_accepts']:.6f}, top scenario share={summary['top_scenario_share_of_false_accepts']:.6f}"
    _write_json(out_dir / "model_scenario_failure_summary.json", summary)
    _plot_bar(model_summary, "model", "accepted_false_accept_count", out_dir / "false_accepts_by_model.png", "Accepted false accepts by model", "count")
    _plot_bar(scenario_summary, "scenario_type", "accepted_false_accept_count", out_dir / "false_accepts_by_scenario.png", "Accepted false accepts by scenario", "count")
    write_model_scenario_report(summary, model_summary, scenario_summary, Path("reports/cstr_model_scenario_failure.md"))
    return summary


def write_model_scenario_report(summary: dict[str, Any], model_summary: pd.DataFrame, scenario_summary: pd.DataFrame, output: Path) -> None:
    text = f"""# CSTR Model and Scenario Failure Audit

## Question

Are CSTR false accepts concentrated in one model or scenario type?

## Model failure summary

{_markdown_table(model_summary, ["model", "accepted_false_accept_count", "accepted_false_accept_rate", "mean_accepted_bad_rmse"])}

## Scenario failure summary

{_markdown_table(scenario_summary, ["scenario_type", "accepted_false_accept_count", "accepted_false_accept_rate", "mean_accepted_bad_rmse"], max_rows=40)}

## Failure concentration

Dominant model: {summary["dominant_failed_model"]}; share {summary["top_model_share_of_false_accepts"]:.6f}.
Dominant scenario type: {summary["dominant_failed_scenario_type"]}; share {summary["top_scenario_share_of_false_accepts"]:.6f}.

## Interpretation

{summary["key_finding"]}

## Verdict

{summary["verdict"]}
"""
    _ensure_dir(output.parent)
    output.write_text(text, encoding="utf-8")


def _far_for_rows(group: pd.DataFrame, judge_id: str, coverage: float, threshold: float) -> dict[str, Any]:
    risk_col = f"risk_{judge_id}"
    accepted = _accepted_mask(group, risk_col, coverage)
    accepted_rows = group.loc[accepted]
    bad = accepted_rows["rmse"].astype(float) > threshold
    return {
        "accepted_count": int(len(accepted_rows)),
        "accepted_false_accept_count": int(bad.sum()),
        "far": float(bad.mean()) if len(accepted_rows) else 0.0,
    }


def _rmse_grid_verdict(grid: pd.DataFrame, config: dict[str, Any]) -> str:
    available = grid[grid["available"]].copy()
    if available.empty:
        return "INCONCLUSIVE"
    low = available[available["coverage"].isin([0.01, 0.02]) & np.isclose(available["threshold"], float(config["bad_rmse_threshold"]))]
    primary = available[available["coverage"].isin([0.05, 0.10]) & np.isclose(available["threshold"], float(config["bad_rmse_threshold"]))]
    if not low.empty and float(low["calibrated_far"].mean()) >= 0.50:
        return "CSTR_ACCEPTED_REGION_TOO_RISKY"
    if not low.empty and not primary.empty:
        low_margin = float(low["absolute_margin"].mean())
        primary_margin = float(primary["absolute_margin"].mean())
        if low_margin >= 0.05 and primary_margin < 0.05 and low_margin >= primary_margin + 0.05:
            return "LOW_COVERAGE_GRANULARITY_PROBLEM"
    practical = available["practical_threshold_passed"].astype(bool)
    positive = available["absolute_margin"].astype(float) > 0
    if positive.mean() >= 0.75 and practical.mean() < 0.50:
        return "THRESHOLD_ROBUST_WEAK_EFFECT"
    margins_by_threshold = available.groupby("threshold")["absolute_margin"].mean()
    if (margins_by_threshold > 0).any() and (margins_by_threshold <= 0).any():
        return "RMSE_TARGET_PROBLEM"
    return "INCONCLUSIVE"


def analyze_cstr_rmse_target(diagnosis_table: str | Path, config_path: str | Path, output: str | Path) -> dict[str, Any]:
    config = load_cstr_weakness_config(config_path)
    table = pd.read_csv(diagnosis_table, low_memory=False)
    base = _base_rows(table)
    out_dir = Path(output)
    _ensure_dir(out_dir)
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    coverages = [float(value) for value in config["granular_coverages"]]
    min_abs = float(config["practical_thresholds"]["minimum_absolute_far_reduction"])
    min_rel = float(config["practical_thresholds"]["minimum_relative_far_reduction"])
    rows = []
    for threshold in thresholds:
        bad = base["rmse"].astype(float) > threshold
        bad_rate = float(bad.mean()) if len(bad) else 0.0
        available = bool(0.0 < bad_rate < 1.0)
        for coverage in coverages:
            baseline_fars = []
            calibrated_fars = []
            accepted_total = 0
            false_total = 0
            for _, group in base.groupby(["model_id", "scenario_type"], sort=False):
                baseline = _far_for_rows(group, BASELINE_JUDGE, coverage, threshold)
                baseline_fars.append(baseline["far"])
                family_results = [_far_for_rows(group, judge, coverage, threshold) for judge in CALIBRATED_FAMILY]
                best = sorted(family_results, key=lambda item: item["far"])[0]
                calibrated_fars.append(best["far"])
                accepted_total += best["accepted_count"]
                false_total += best["accepted_false_accept_count"]
            baseline_far = float(np.mean(baseline_fars)) if baseline_fars else 0.0
            calibrated_far = float(np.mean(calibrated_fars)) if calibrated_fars else 0.0
            margin = baseline_far - calibrated_far
            relative = float(margin / baseline_far) if baseline_far > 0 else 0.0
            rows.append(
                {
                    "threshold": float(threshold),
                    "coverage": float(coverage),
                    "bad_rate": bad_rate,
                    "available": available,
                    "baseline_far": baseline_far,
                    "calibrated_far": calibrated_far,
                    "absolute_margin": margin,
                    "relative_margin": relative,
                    "accepted_count": int(accepted_total),
                    "accepted_false_accept_count": int(false_total),
                    "practical_threshold_passed": bool(available and (margin >= min_abs or relative >= min_rel)),
                }
            )
    grid = pd.DataFrame(rows)
    grid.to_csv(out_dir / "rmse_threshold_coverage_grid.csv", index=False)
    verdict = _rmse_grid_verdict(grid, config)
    useful = grid[grid["available"] & grid["practical_threshold_passed"]]
    failed = grid[grid["available"] & ~grid["practical_threshold_passed"]]
    summary = {
        "verdict": verdict,
        "key_finding": f"{len(useful)} threshold/coverage cells pass practical thresholds; {len(failed)} available cells do not",
        "useful_region_count": int(len(useful)),
        "failed_region_count": int(len(failed)),
        "degenerate_region_count": int((~grid["available"]).sum()),
        "best_margin": float(grid["absolute_margin"].max()) if len(grid) else 0.0,
        "best_margin_threshold": float(grid.loc[grid["absolute_margin"].idxmax(), "threshold"]) if len(grid) else 0.0,
        "best_margin_coverage": float(grid.loc[grid["absolute_margin"].idxmax(), "coverage"]) if len(grid) else 0.0,
    }
    _write_json(out_dir / "rmse_target_summary.json", summary)
    _plot_rmse_grid(grid, out_dir / "cstr_far_by_threshold_coverage.png")
    write_rmse_target_report(summary, grid, Path("reports/cstr_rmse_target_audit.md"))
    return summary


def _plot_rmse_grid(grid: pd.DataFrame, output: Path) -> None:
    _ensure_dir(output.parent)
    pivot = grid.pivot(index="threshold", columns="coverage", values="absolute_margin")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="coolwarm")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{c:g}" for c in pivot.columns], rotation=45)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"{t:g}" for t in pivot.index])
    ax.set_xlabel("coverage")
    ax.set_ylabel("RMSE threshold")
    ax.set_title("CSTR FAR margin by threshold/coverage")
    fig.colorbar(im, ax=ax, label="baseline FAR - calibrated FAR")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def write_rmse_target_report(summary: dict[str, Any], grid: pd.DataFrame, output: Path) -> None:
    low = grid[grid["coverage"].isin([0.01, 0.02, 0.05, 0.10])].copy()
    low = low.rename(columns={"absolute_margin": "margin"})
    text = f"""# CSTR RMSE Target and Threshold Sensitivity Audit

## Question

Is CSTR weakness caused by the chosen RMSE target or threshold?

## Threshold/coverage grid

{grid.shape[0]} threshold/coverage cells evaluated.

## Low-coverage granularity

{_markdown_table(low, ["coverage", "threshold", "baseline_far", "calibrated_far", "margin", "accepted_count"], max_rows=80)}

## Regions where calibrated judge is useful

{summary["useful_region_count"]}

## Regions where calibrated judge fails

{summary["failed_region_count"]}

## Interpretation

{summary["key_finding"]}

## Verdict

{summary["verdict"]}
"""
    _ensure_dir(output.parent)
    output.write_text(text, encoding="utf-8")


def _load_required_summary(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    return _load_json(path)


def _final_diagnosis(summaries: dict[str, dict[str, Any]]) -> tuple[str, str]:
    repair = summaries["repair_signal"]
    statewise = summaries["statewise_error"]
    signal = summaries["signal_overlap"]
    model = summaries["model_scenario_failure"]
    rmse = summaries["rmse_target"]
    if repair.get("verdict") == "REPAIR_SIGNAL_BLIND_SPOT" and float(repair.get("fraction_accepted_false_accepts_with_low_repair", 0.0)) >= 0.50:
        return "REPAIR_SIGNAL_BLIND_SPOT", "FIX_REPAIR_SIGNAL"
    if model.get("verdict") == "MODEL_SPECIFIC_CSTR_FAILURE":
        return "MODEL_SPECIFIC_CSTR_FAILURE", "FOCUS_ON_MODEL_FAILURE"
    if model.get("verdict") == "SCENARIO_SPECIFIC_CSTR_FAILURE":
        return "SCENARIO_SPECIFIC_CSTR_FAILURE", "FOCUS_ON_SCENARIO_FAILURE"
    if statewise.get("verdict") in {"TEMPERATURE_DOMINATED", "CONCENTRATION_DOMINATED"}:
        return "STATE_SPECIFIC_CSTR_FAILURE", "NARROW_CLAIM_ONLY"
    if rmse.get("verdict") == "RMSE_TARGET_PROBLEM":
        return "RMSE_TARGET_PROBLEM", "CHANGE_TARGET_TO_EVENT_RISK"
    if rmse.get("verdict") == "CSTR_ACCEPTED_REGION_TOO_RISKY":
        return "CSTR_ACCEPTED_REGION_TOO_RISKY", "DO_NOT_EXPAND"
    if rmse.get("verdict") == "LOW_COVERAGE_GRANULARITY_PROBLEM":
        return "LOW_COVERAGE_GRANULARITY_PROBLEM", "NARROW_CLAIM_ONLY"
    if signal.get("verdict") == "SIGNAL_BLIND_SPOT" and model.get("verdict") == "DIFFUSE_CSTR_FAILURE":
        return "DIFFUSE_SIGNAL_BLIND_SPOT", "DO_NOT_EXPAND"
    return "INCONCLUSIVE", "NARROW_CLAIM_ONLY"


def make_cstr_weakness_diagnosis(input_dir: str | Path, output: str | Path) -> dict[str, Any]:
    root = Path(input_dir)
    summaries = {
        "preconditions": _load_required_summary(root / "preconditions" / "precondition_check.json"),
        "diagnosis_table": _load_required_summary(root / "diagnosis_table" / "cstr_diagnosis_schema.json"),
        "statewise_error": _load_required_summary(root / "statewise_error" / "statewise_error_summary.json"),
        "repair_signal": _load_required_summary(root / "repair_signal" / "repair_signal_metrics.json"),
        "signal_overlap": _load_required_summary(root / "signal_overlap" / "signal_separability_metrics.json"),
        "model_scenario_failure": _load_required_summary(root / "model_scenario_failure" / "model_scenario_failure_summary.json"),
        "rmse_target": _load_required_summary(root / "rmse_target" / "rmse_target_summary.json"),
    }
    diagnosis, next_action = _final_diagnosis(summaries)
    result = {
        "final_diagnosis": diagnosis,
        "recommended_next_action": next_action,
        "expansion_allowed": False,
        "allowed_claim": "A weak but positive low-coverage result under the frozen protocol.",
        "forbidden_claims": [
            "strong two-system support",
            "safety certification",
            "product readiness",
            "general reliable counterfactual simulation",
        ],
        "analysis_verdicts": {key: value.get("verdict") for key, value in summaries.items()},
    }
    output_path = Path(output)
    _ensure_dir(output_path.parent)
    _write_json(output_path.with_suffix(".json"), result)
    write_final_diagnosis_report(result, summaries, output_path)
    return result


def write_final_diagnosis_report(result: dict[str, Any], summaries: dict[str, dict[str, Any]], output: Path) -> None:
    rows = []
    for key, summary in summaries.items():
        rows.append({"analysis": key, "verdict": summary.get("verdict", ""), "key finding": summary.get("key_finding", "")})
    evidence = pd.DataFrame(rows)
    text = f"""# CSTR Weakness Diagnosis

## Starting point

The current controlling claim is weak low-coverage support. Expansion is forbidden.

## Evidence summary

{_markdown_table(evidence, ["analysis", "verdict", "key finding"])}

## Final diagnosis

{result["final_diagnosis"]}

## Explanation

Diagnosis follows the fixed hierarchy in the CSTR weakness audit plan. The weak practical CSTR effect is not promoted to a broad claim.

## What not to do next

Do not add RSSM, a third system, heat exchanger evidence, a new judge, a new model, product/API/UI work, or a paper draft as a substitute for diagnosing CSTR weakness.

## Recommended next action

{result["recommended_next_action"]}

## Allowed claims after this diagnosis

{result["allowed_claim"]}

## Forbidden claims

{", ".join(result["forbidden_claims"])}
"""
    output.write_text(text, encoding="utf-8")
