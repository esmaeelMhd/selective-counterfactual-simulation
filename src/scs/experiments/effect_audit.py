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


SYSTEM_RESULT_DIRS = {
    "two_tank": Path("results/calibrated_two_tank"),
    "cstr": Path("results/calibrated_cstr"),
}
SYSTEM_SEED_DIRS = {
    "two_tank": Path("results/calibrated_seed_sweep_two_tank"),
    "cstr": Path("results/calibrated_seed_sweep_cstr"),
}
SYSTEM_STRESS_DIRS = {
    "two_tank": Path("results/calibrated_stress_two_tank"),
    "cstr": Path("results/calibrated_stress_cstr"),
}
SYSTEM_CONFIGS = {
    "two_tank": Path("configs/experiments/calibrated_two_tank.yaml"),
    "cstr": Path("configs/experiments/calibrated_cstr.yaml"),
}
SIGNAL_COLUMNS = [
    "support_distance",
    "uncertainty_score",
    "disagreement_score",
    "invariant_residual",
    "repair_amount",
]
REQUIRED_PRIOR_ARTIFACTS = [
    "results/calibrated_two_tank/calibrated_judge_summary.json",
    "results/calibrated_two_tank/calibrated_risk_coverage.csv",
    "results/calibrated_two_tank/test_table.csv",
    "results/calibrated_seed_sweep_two_tank/seed_sweep_calibrated_summary.json",
    "results/calibrated_seed_sweep_two_tank/calibrated_risk_coverage_all.csv",
    "results/calibrated_stress_two_tank/stress_summary.json",
    "results/calibrated_cstr/calibrated_judge_summary.json",
    "results/calibrated_cstr/calibrated_risk_coverage.csv",
    "results/calibrated_cstr/test_table.csv",
    "results/calibrated_seed_sweep_cstr/seed_sweep_calibrated_summary.json",
    "results/calibrated_seed_sweep_cstr/calibrated_risk_coverage_all.csv",
    "results/calibrated_stress_cstr/stress_summary.json",
    "results/cstr_sanity/cstr_label_checks.json",
    "reports/multi_system_calibrated_decision_gate.md",
    "reports/multi_system_calibrated_decision_gate.json",
    "docs/calibrated_protocol_lock_v1.md",
]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_yaml(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return data


def _markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        if not df.empty:
            raise KeyError(f"missing columns for markdown table: {missing_columns}")
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


def load_effect_audit_config(path: str | Path) -> dict[str, Any]:
    config = _load_yaml(path)
    required = {
        "audit_id",
        "systems",
        "primary_coverages",
        "baseline_judge",
        "calibrated_family",
        "diagnostic_oracle",
        "minimum_absolute_far_reduction",
        "minimum_relative_far_reduction",
        "bootstrap_iterations",
        "confidence_level",
        "seed_win_threshold_strong",
        "seed_win_threshold_weak",
        "bad_rmse_thresholds",
        "forbidden",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"missing effect audit config keys: {missing}")
    if config["systems"] != ["two_tank", "cstr"]:
        raise ValueError("effect audit systems must be exactly ['two_tank', 'cstr']")
    forbidden = config["forbidden"]
    for key in ["allow_new_judges", "allow_new_models", "allow_new_systems", "allow_protocol_mutation"]:
        if forbidden.get(key) is not False:
            raise ValueError(f"forbidden.{key} must be false")
    return config


def _primary_threshold(system_id: str) -> float:
    risk = pd.read_csv(SYSTEM_RESULT_DIRS[system_id] / "calibrated_risk_coverage.csv", usecols=["threshold"])
    values = sorted(float(value) for value in risk["threshold"].unique())
    if len(values) != 1:
        raise ValueError(f"expected one primary threshold for {system_id}, got {values}")
    return values[0]


def _best_calibrated_judge(system_id: str, coverage: float) -> str:
    low = pd.read_csv(SYSTEM_RESULT_DIRS[system_id] / "low_coverage_summary.csv")
    row = low[np.isclose(low["coverage"], float(coverage))]
    if row.empty:
        raise ValueError(f"missing low-coverage row for {system_id} coverage {coverage}")
    return str(row.iloc[0]["best_calibrated_judge"])


def _accepted_mask_for_group(group: pd.DataFrame, judge_id: str, coverage: float) -> pd.Series:
    risk_col = f"risk_{judge_id}"
    if risk_col not in group.columns:
        raise ValueError(f"missing risk column: {risk_col}")
    accepted_count = min(max(int(math.ceil(float(coverage) * len(group))), 1), len(group))
    accepted_index = group.sort_values(risk_col, kind="mergesort").head(accepted_count).index
    mask = pd.Series(False, index=group.index)
    mask.loc[accepted_index] = True
    return mask


def far_for_table(table: pd.DataFrame, judge_id: str, coverage: float, bad_column: str = "bad_rmse_label") -> dict[str, float]:
    fars = []
    accepted_total = 0
    false_total = 0
    for _, group in table.groupby(["model_id", "scenario_type"], sort=False):
        accepted = _accepted_mask_for_group(group, judge_id, coverage)
        accepted_rows = group.loc[accepted]
        bad = accepted_rows[bad_column].astype(bool)
        accepted_total += int(len(accepted_rows))
        false_total += int(bad.sum())
        fars.append(float(bad.mean()) if len(accepted_rows) else 0.0)
    return {
        "false_accept_rate": float(np.mean(fars)) if fars else 0.0,
        "accepted_count": float(accepted_total),
        "false_accept_count": float(false_total),
    }


def family_best_far_for_table(
    table: pd.DataFrame,
    family: list[str],
    coverage: float,
    bad_column: str = "bad_rmse_label",
) -> dict[str, Any]:
    group_rows = []
    for key, group in table.groupby(["model_id", "scenario_type"], sort=False):
        judge_rows = []
        for judge_id in family:
            result = far_for_table(group, judge_id, coverage, bad_column=bad_column)
            judge_rows.append(
                {
                    "model_id": key[0],
                    "scenario_type": key[1],
                    "judge_id": judge_id,
                    "false_accept_rate": result["false_accept_rate"],
                    "accepted_count": result["accepted_count"],
                    "false_accept_count": result["false_accept_count"],
                }
            )
        best = sorted(judge_rows, key=lambda row: (row["false_accept_rate"], row["judge_id"]))[0]
        group_rows.append(best)
    frame = pd.DataFrame(group_rows)
    return {
        "false_accept_rate": float(frame["false_accept_rate"].mean()) if not frame.empty else 0.0,
        "accepted_count": float(frame["accepted_count"].sum()) if not frame.empty else 0.0,
        "false_accept_count": float(frame["false_accept_count"].sum()) if not frame.empty else 0.0,
        "group_judges": frame,
    }


def _bootstrap_margin_samples(
    table: pd.DataFrame,
    baseline_judge: str,
    calibrated_family: list[str],
    coverage: float,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    groups = []
    for _, group in table.groupby(["model_id", "scenario_type"], sort=False):
        group_data = {
            "labels": group["bad_rmse_label"].astype(bool).to_numpy(),
            "baseline_risk": group[f"risk_{baseline_judge}"].to_numpy(dtype=float),
            "family_risk": {
                judge: group[f"risk_{judge}"].to_numpy(dtype=float)
                for judge in calibrated_family
            },
        }
        groups.append(group_data)
    rows = []
    for iteration in range(iterations):
        baseline_group_fars = []
        calibrated_group_fars = []
        for group in groups:
            labels = group["labels"]
            take = rng.integers(0, len(labels), size=len(labels))
            sampled_labels = labels[take]
            accepted_count = min(max(int(math.ceil(float(coverage) * len(sampled_labels))), 1), len(sampled_labels))
            baseline_risk = group["baseline_risk"][take]
            baseline_order = np.argsort(baseline_risk, kind="mergesort")[:accepted_count]
            baseline_group_fars.append(float(np.mean(sampled_labels[baseline_order])))
            family_fars = []
            for judge in calibrated_family:
                risk = group["family_risk"][judge][take]
                order = np.argsort(risk, kind="mergesort")[:accepted_count]
                family_fars.append(float(np.mean(sampled_labels[order])))
            calibrated_group_fars.append(float(np.min(family_fars)))
        baseline = float(np.mean(baseline_group_fars)) if baseline_group_fars else 0.0
        calibrated = float(np.mean(calibrated_group_fars)) if calibrated_group_fars else 0.0
        rows.append(
            {
                "bootstrap_iteration": iteration,
                "coverage": float(coverage),
                "baseline_far": baseline,
                "calibrated_far": calibrated,
                "absolute_margin": baseline - calibrated,
            }
        )
    return pd.DataFrame(rows)


def _ci(values: pd.Series | np.ndarray, confidence: float) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if len(array) == 0:
        return (0.0, 0.0)
    alpha = 1.0 - confidence
    return (float(np.quantile(array, alpha / 2.0)), float(np.quantile(array, 1.0 - alpha / 2.0)))


def _paired_seed_ci(margins: pd.Series, confidence: float) -> tuple[float, float]:
    values = np.asarray(margins, dtype=float)
    if len(values) == 0:
        return (0.0, 0.0)
    mean = float(np.mean(values))
    if len(values) == 1 or float(np.std(values, ddof=1)) <= 1e-12:
        constant = float(values[0])
        return (constant, constant)
    half_width = float(stats.t.ppf(0.5 + confidence / 2.0, len(values) - 1) * stats.sem(values))
    return (mean - half_width, mean + half_width)


def _seed_level_margins(system_id: str, config: dict[str, Any], coverage: float) -> pd.DataFrame:
    risk = pd.read_csv(SYSTEM_SEED_DIRS[system_id] / "calibrated_risk_coverage_all.csv")
    risk = risk[np.isclose(risk["coverage_requested"], float(coverage))]
    baseline_judge = str(config["baseline_judge"])
    family = [str(value) for value in config["calibrated_family"]]
    rows = []
    for seed, seed_frame in risk.groupby("seed", sort=True):
        baseline = seed_frame[seed_frame["judge_id"] == baseline_judge]["false_accept_rate"].mean()
        calibrated = (
            seed_frame[seed_frame["judge_id"].isin(family)]
            .groupby(["model_id", "scenario_type"])["false_accept_rate"]
            .min()
            .mean()
        )
        rows.append(
            {
                "system_id": system_id,
                "seed": int(seed),
                "coverage": float(coverage),
                "baseline_far": float(baseline),
                "best_calibrated_family_far": float(calibrated),
                "absolute_margin": float(baseline - calibrated),
                "relative_margin": float((baseline - calibrated) / baseline) if baseline > 0 else 0.0,
                "win": bool(calibrated < baseline - 1e-12),
            }
        )
    return pd.DataFrame(rows)


def _row_verdict(row: pd.Series) -> str:
    if row["absolute_margin"] <= 0 or row["seed_win_rate"] < row["seed_win_threshold_weak"]:
        return "NOT_SUPPORTED"
    if row["ci_excludes_zero"] and (row["meets_absolute_threshold"] or row["meets_relative_threshold"]):
        return "PRACTICALLY_MEANINGFUL"
    return "POSITIVE_BUT_WEAK"


def _effect_verdict(rows: pd.DataFrame, config: dict[str, Any]) -> str:
    meaningful_systems = set(rows[rows["verdict"] == "PRACTICALLY_MEANINGFUL"]["system_id"])
    positive_systems = set(rows[rows["absolute_margin"] > 0]["system_id"])
    weak_seed = rows.groupby("system_id")["seed_win_rate"].max() < float(config["seed_win_threshold_weak"])
    if bool(weak_seed.any()):
        return "NOT_SUPPORTED"
    if meaningful_systems == {"two_tank", "cstr"}:
        strong_seed = rows.groupby("system_id")["seed_win_rate"].max() >= float(config["seed_win_threshold_strong"])
        if bool(strong_seed.all()):
            return "STRONG_TWO_SYSTEM_EFFECT"
    if positive_systems == {"two_tank", "cstr"}:
        return "WEAK_TWO_SYSTEM_EFFECT"
    if len(meaningful_systems) == 1 and len(positive_systems) == 2:
        return "MIXED_EFFECT"
    return "NOT_SUPPORTED"


def verify_effect_size_audit_preconditions(config_path: str | Path, output: str | Path) -> dict[str, Any]:
    config = load_effect_audit_config(config_path)
    out_dir = Path(output)
    _ensure_dir(out_dir)
    missing = [path for path in REQUIRED_PRIOR_ARTIFACTS if not Path(path).exists() or Path(path).stat().st_size == 0]
    prior = {
        "two_tank_single": _load_json("results/calibrated_two_tank/calibrated_judge_summary.json") if not missing else {},
        "two_tank_seed": _load_json("results/calibrated_seed_sweep_two_tank/seed_sweep_calibrated_summary.json") if not missing else {},
        "two_tank_stress": _load_json("results/calibrated_stress_two_tank/stress_summary.json") if not missing else {},
        "cstr_sanity": _load_json("results/cstr_sanity/cstr_label_checks.json") if not missing else {},
        "cstr_single": _load_json("results/calibrated_cstr/calibrated_judge_summary.json") if not missing else {},
        "cstr_seed": _load_json("results/calibrated_seed_sweep_cstr/seed_sweep_calibrated_summary.json") if not missing else {},
        "cstr_stress": _load_json("results/calibrated_stress_cstr/stress_summary.json") if not missing else {},
        "multi_gate": _load_json("reports/multi_system_calibrated_decision_gate.json") if not missing else {},
    }
    expected = {
        "two_tank_single": ("verdict", "SUPPORTED_LOW_COVERAGE"),
        "two_tank_seed": ("verdict", "ROBUST_LOW_COVERAGE"),
        "two_tank_stress": ("verdict", "ROBUST_LOW_COVERAGE_ONLY"),
        "cstr_sanity": ("verdict", "VALID_CSTR_BENCHMARK"),
        "cstr_single": ("verdict", "SUPPORTED_LOW_COVERAGE"),
        "cstr_seed": ("verdict", "ROBUST_LOW_COVERAGE"),
        "cstr_stress": ("verdict", "ROBUST_LOW_COVERAGE_ONLY"),
        "multi_gate": ("decision", "TWO_SYSTEM_LOW_COVERAGE_SUPPORTED"),
    }
    verdict_mismatches = []
    for key, (field, expected_value) in expected.items():
        if prior.get(key, {}).get(field) != expected_value:
            verdict_mismatches.append({"artifact": key, "expected": expected_value, "actual": prior.get(key, {}).get(field)})
    leakage = any(
        bool(prior.get(key, {}).get("leakage_detected"))
        for key in ["two_tank_single", "two_tank_seed", "two_tank_stress", "cstr_single", "cstr_seed", "cstr_stress", "multi_gate"]
    )
    oracle_violations = []
    for system_id in config["systems"]:
        risk = pd.read_csv(SYSTEM_RESULT_DIRS[system_id] / "calibrated_risk_coverage.csv") if not missing else pd.DataFrame()
        oracle = risk[risk["judge_id"] == config["diagnostic_oracle"]] if not risk.empty else pd.DataFrame()
        if oracle.empty or not bool(oracle["is_oracle"].all()) or bool(oracle["is_real_judge"].any()):
            oracle_violations.append(system_id)
    protocol_lock = Path("docs/calibrated_protocol_lock_v1.md").exists()
    reasons = []
    if missing:
        reasons.append("missing required artifacts")
    if verdict_mismatches:
        reasons.append("prior verdict mismatch")
    if leakage:
        reasons.append("leakage detected in prior evidence")
    if oracle_violations:
        reasons.append("oracle is not diagnostic-only in risk coverage")
    if not protocol_lock:
        reasons.append("protocol lock missing")
    verdict = "READY_FOR_EFFECT_SIZE_AUDIT" if not reasons else "NOT_READY"
    result = {
        "required_artifacts": {path: Path(path).exists() and Path(path).stat().st_size > 0 for path in REQUIRED_PRIOR_ARTIFACTS},
        "missing_artifacts": missing,
        "prior_verdicts": {key: prior.get(key, {}).get("verdict") or prior.get(key, {}).get("decision") for key in prior},
        "verdict_mismatches": verdict_mismatches,
        "leakage_detected": leakage,
        "oracle_violations": oracle_violations,
        "protocol_lock_exists": protocol_lock,
        "practical_threshold_config": config,
        "verdict": verdict,
        "reasons": reasons,
    }
    _write_json(out_dir / "precondition_check.json", result)
    write_precondition_report(result, Path("reports/effect_size_audit_precondition_check.md"))
    if verdict != "READY_FOR_EFFECT_SIZE_AUDIT":
        raise RuntimeError(f"effect-size audit preconditions not ready: {reasons}")
    return result


def write_precondition_report(result: dict[str, Any], output: Path) -> None:
    _ensure_dir(output.parent)
    artifacts = pd.DataFrame(
        [{"artifact": key, "exists": value} for key, value in result["required_artifacts"].items()]
    )
    verdicts = pd.DataFrame(
        [{"artifact": key, "verdict": value} for key, value in result["prior_verdicts"].items()]
    )
    text = f"""# Effect-Size Audit Preconditions

## Required artifacts

{_markdown_table(artifacts, ["artifact", "exists"])}

## Prior verdicts

{_markdown_table(verdicts, ["artifact", "verdict"])}

## Leakage status

{result["leakage_detected"]}

## Protocol lock status

{result["protocol_lock_exists"]}

## Practical threshold config

minimum_absolute_far_reduction: {result["practical_threshold_config"]["minimum_absolute_far_reduction"]}
minimum_relative_far_reduction: {result["practical_threshold_config"]["minimum_relative_far_reduction"]}
confidence_level: {result["practical_threshold_config"]["confidence_level"]}

## Verdict

{result["verdict"]}
"""
    output.write_text(text, encoding="utf-8")


def analyze_effect_size_uncertainty(config_path: str | Path, output: str | Path) -> dict[str, Any]:
    config = load_effect_audit_config(config_path)
    out_dir = Path(output)
    _ensure_dir(out_dir)
    all_bootstrap = []
    all_seed = []
    rows = []
    confidence = float(config["confidence_level"])
    iterations = int(config["bootstrap_iterations"])
    for system_idx, system_id in enumerate(config["systems"]):
        table = pd.read_csv(SYSTEM_RESULT_DIRS[system_id] / "test_table.csv")
        threshold = _primary_threshold(system_id)
        baseline_judge = str(config["baseline_judge"])
        calibrated_family = [str(value) for value in config["calibrated_family"]]
        for coverage in [float(value) for value in config["primary_coverages"]]:
            baseline = far_for_table(table, baseline_judge, coverage)["false_accept_rate"]
            calibrated_result = family_best_far_for_table(table, calibrated_family, coverage)
            calibrated = calibrated_result["false_accept_rate"]
            margin = baseline - calibrated
            relative = margin / baseline if baseline > 0 else 0.0
            bootstrap = _bootstrap_margin_samples(
                table,
                baseline_judge,
                calibrated_family,
                coverage,
                iterations,
                seed=9100 + 100 * system_idx + int(round(coverage * 1000)),
            )
            bootstrap.insert(0, "system_id", system_id)
            bootstrap["calibrated_judge"] = "best_calibrated_family_per_model_scenario"
            all_bootstrap.append(bootstrap)
            boot_low, boot_high = _ci(bootstrap["absolute_margin"], confidence)
            seed_frame = _seed_level_margins(system_id, config, coverage)
            seed_frame["calibrated_judge_policy"] = "best_calibrated_family_per_model_scenario"
            all_seed.append(seed_frame)
            seed_low, seed_high = _paired_seed_ci(seed_frame["absolute_margin"], confidence)
            seed_margin_mean = float(seed_frame["absolute_margin"].mean())
            seed_win_rate = float(seed_frame["win"].mean())
            row = {
                "system_id": system_id,
                "coverage": coverage,
                "baseline_judge": baseline_judge,
                "calibrated_judge": "best_calibrated_family_per_model_scenario",
                "baseline_far": baseline,
                "calibrated_far": calibrated,
                "absolute_margin": margin,
                "relative_margin": relative,
                "bootstrap_ci_low": boot_low,
                "bootstrap_ci_high": boot_high,
                "seed_margin_mean": seed_margin_mean,
                "seed_ci_low": seed_low,
                "seed_ci_high": seed_high,
                "seed_win_rate": seed_win_rate,
                "meets_absolute_threshold": bool(margin >= float(config["minimum_absolute_far_reduction"])),
                "meets_relative_threshold": bool(relative >= float(config["minimum_relative_far_reduction"])),
                "ci_excludes_zero": bool(boot_low > 0.0 and seed_low > 0.0),
                "seed_win_threshold_weak": float(config["seed_win_threshold_weak"]),
            }
            row["verdict"] = _row_verdict(pd.Series(row))
            rows.append(row)
    effect = pd.DataFrame(rows)
    effect.to_csv(out_dir / "effect_size_by_system.csv", index=False)
    bootstrap_samples = pd.concat(all_bootstrap, ignore_index=True)
    bootstrap_samples.to_csv(out_dir / "bootstrap_samples.csv", index=False)
    seed_margins = pd.concat(all_seed, ignore_index=True)
    seed_margins.to_csv(out_dir / "seed_level_margins.csv", index=False)
    verdict = _effect_verdict(effect, config)
    summary = {
        "verdict": verdict,
        "minimum_absolute_far_reduction": float(config["minimum_absolute_far_reduction"]),
        "minimum_relative_far_reduction": float(config["minimum_relative_far_reduction"]),
        "confidence_level": confidence,
        "bootstrap_iterations": iterations,
        "rows": effect.to_dict(orient="records"),
        "known_limitations": [
            "Bootstrap samples scenario-model rows within each model/scenario group.",
            "Seed-level calibrated margin uses the frozen calibrated family outputs from each seed sweep.",
            "The audit quantifies existing evidence and does not tune the judge.",
        ],
    }
    _write_json(out_dir / "effect_size_summary.json", summary)
    plot_effect_size_ci(effect, out_dir / "effect_size_ci.png")
    write_effect_size_report(config, summary, effect, Path("reports/effect_size_and_uncertainty_audit.md"))
    return summary


def plot_effect_size_ci(effect: pd.DataFrame, output: Path) -> None:
    _ensure_dir(output.parent)
    labels = [f"{row.system_id}\n{row.coverage:g}" for row in effect.itertuples()]
    x = np.arange(len(effect))
    y = effect["absolute_margin"].to_numpy(dtype=float)
    low = y - effect["bootstrap_ci_low"].to_numpy(dtype=float)
    high = effect["bootstrap_ci_high"].to_numpy(dtype=float) - y
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(x, y, yerr=np.vstack([low, high]), fmt="o", capsize=4, label="scenario bootstrap CI")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.axhline(0.05, color="tab:orange", linestyle="--", linewidth=1, label="absolute threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("FAR margin")
    ax.set_xlabel("system / coverage")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def write_effect_size_report(config: dict[str, Any], summary: dict[str, Any], effect: pd.DataFrame, output: Path) -> None:
    _ensure_dir(output.parent)
    table = effect.copy()
    table["bootstrap_95_ci"] = table.apply(lambda row: f"[{row.bootstrap_ci_low:.6f}, {row.bootstrap_ci_high:.6f}]", axis=1)
    table["seed_95_ci"] = table.apply(lambda row: f"[{row.seed_ci_low:.6f}, {row.seed_ci_high:.6f}]", axis=1)
    table["practical_threshold_passed"] = table["meets_absolute_threshold"] | table["meets_relative_threshold"]
    text = f"""# Effect Size and Uncertainty Audit

## Question

Is the calibrated low-coverage improvement statistically and practically meaningful?

## Fixed practical thresholds

minimum_absolute_far_reduction: {config["minimum_absolute_far_reduction"]}
minimum_relative_far_reduction: {config["minimum_relative_far_reduction"]}
confidence_level: {config["confidence_level"]}
bootstrap_iterations: {config["bootstrap_iterations"]}

## Main result

{_markdown_table(table.rename(columns={"system_id": "system"}), ["system", "coverage", "absolute_margin", "relative_margin", "bootstrap_95_ci", "seed_95_ci", "seed_win_rate", "practical_threshold_passed", "verdict"])}

## TwoTank interpretation

{_system_effect_text(effect, "two_tank")}

## CSTR interpretation

{_system_effect_text(effect, "cstr")}

## Cross-system interpretation

The audit keeps the TwoTank and CSTR margins separate. A small CSTR margin is not promoted to a strong practical effect.

## Verdict

{summary["verdict"]}

## Explanation

The final verdict follows fixed practical thresholds from `configs/audits/effect_size_audit.yaml`.

## Known limitations

{"; ".join(summary["known_limitations"])}

## Reproduction command

```bash
python scripts/analyze_effect_size_uncertainty.py --config configs/audits/effect_size_audit.yaml --output results/effect_size_audit/effect_size
```
"""
    output.write_text(text, encoding="utf-8")


def _system_effect_text(effect: pd.DataFrame, system_id: str) -> str:
    rows = effect[effect["system_id"] == system_id]
    parts = []
    for row in rows.itertuples():
        parts.append(
            f"coverage {row.coverage:g}: margin {row.absolute_margin:.6f}, "
            f"relative {row.relative_margin:.6f}, seed win rate {row.seed_win_rate:.6f}, verdict {row.verdict}"
        )
    return "\n".join(f"- {part}" for part in parts)


def _accepted_rows(table: pd.DataFrame, judge_id: str, coverage: float) -> pd.DataFrame:
    frames = []
    for _, group in table.groupby(["model_id", "scenario_type"], sort=False):
        risk_col = f"risk_{judge_id}"
        ordered = group.sort_values(risk_col, kind="mergesort").copy()
        accepted_count = min(max(int(math.ceil(float(coverage) * len(ordered))), 1), len(ordered))
        accepted = ordered.head(accepted_count).copy()
        accepted["risk_score"] = accepted[risk_col].astype(float)
        accepted["rank_within_accepted"] = np.arange(1, len(accepted) + 1)
        frames.append(accepted)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _accepted_rows_for_family_best(table: pd.DataFrame, family: list[str], coverage: float) -> pd.DataFrame:
    frames = []
    for _, group in table.groupby(["model_id", "scenario_type"], sort=False):
        judge_rows = []
        for judge_id in family:
            result = far_for_table(group, judge_id, coverage)
            judge_rows.append((result["false_accept_rate"], judge_id))
        selected_judge = sorted(judge_rows, key=lambda item: (item[0], item[1]))[0][1]
        accepted = _accepted_rows(group, selected_judge, coverage)
        accepted["judge_id"] = selected_judge
        frames.append(accepted)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _tag_false_accepts(false_accepts: pd.DataFrame, source_table: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if false_accepts.empty:
        false_accepts["tags"] = []
        return false_accepts
    quantiles = {signal: float(source_table[signal].quantile(0.25)) for signal in SIGNAL_COLUMNS}
    model_counts = false_accepts["model_id"].value_counts(normalize=True)
    scenario_counts = false_accepts["scenario_type"].value_counts(normalize=True)
    dominant_models = set(model_counts[model_counts >= 0.40].index)
    dominant_scenarios = set(scenario_counts[scenario_counts >= 0.40].index)
    tagged = []
    for row in false_accepts.to_dict(orient="records"):
        tags = []
        if row["support_distance"] <= quantiles["support_distance"]:
            tags.append("LOW_SUPPORT_RISK_BUT_BAD")
        if row["uncertainty_score"] <= quantiles["uncertainty_score"]:
            tags.append("LOW_UNCERTAINTY_BUT_BAD")
        if row["disagreement_score"] <= quantiles["disagreement_score"]:
            tags.append("LOW_DISAGREEMENT_BUT_BAD")
        if row["invariant_residual"] <= quantiles["invariant_residual"]:
            tags.append("LOW_INVARIANT_RESIDUAL_BUT_BAD")
        if row["repair_amount"] <= quantiles["repair_amount"]:
            tags.append("LOW_REPAIR_BUT_BAD")
        if row["model_id"] in dominant_models:
            tags.append("MODEL_SPECIFIC_FAILURE")
        if row["scenario_type"] in dominant_scenarios:
            tags.append("SPLIT_SPECIFIC_FAILURE")
        if float(row["rmse"]) <= threshold * 1.25:
            tags.append("NEAR_THRESHOLD_FAILURE")
        if float(row["rmse"]) >= threshold * 3.0:
            tags.append("SEVERE_MISCLASSIFICATION")
        row["tags"] = ";".join(tags) if tags else "UNSPECIFIED"
        tagged.append(row)
    return pd.DataFrame(tagged)


def analyze_accepted_false_accepts(config_path: str | Path, output: str | Path) -> dict[str, Any]:
    config = load_effect_audit_config(config_path)
    out_dir = Path(output)
    _ensure_dir(out_dir)
    all_false_accepts = []
    system_summaries = []
    for system_id in config["systems"]:
        table = pd.read_csv(SYSTEM_RESULT_DIRS[system_id] / "test_table.csv")
        threshold = _primary_threshold(system_id)
        family = [str(value) for value in config["calibrated_family"]]
        for coverage in [float(value) for value in config["primary_coverages"]]:
            accepted = _accepted_rows_for_family_best(table, family, coverage)
            false_accepts = accepted[accepted["bad_rmse_label"].astype(bool)].copy()
            false_accepts = _tag_false_accepts(false_accepts, table, threshold)
            false_accepts["coverage"] = coverage
            all_false_accepts.append(false_accepts)
            system_summaries.append(
                {
                    "system_id": system_id,
                    "coverage": coverage,
                    "judge_id": "best_calibrated_family_per_model_scenario",
                    "accepted_count": int(len(accepted)),
                    "accepted_bad_count": int(len(false_accepts)),
                    "accepted_bad_rate": float(len(false_accepts) / len(accepted)) if len(accepted) else 0.0,
                }
            )
    accepted_false_accepts = pd.concat(all_false_accepts, ignore_index=True) if all_false_accepts else pd.DataFrame()
    columns = [
        "system_id",
        "scenario_id",
        "model_id",
        "scenario_type",
        "coverage",
        "judge_id",
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
        "rank_within_accepted",
        "tags",
    ]
    accepted_false_accepts = accepted_false_accepts[columns] if not accepted_false_accepts.empty else pd.DataFrame(columns=columns)
    accepted_false_accepts.to_csv(out_dir / "accepted_false_accepts.csv", index=False)
    by_split = _false_accept_group_summary(accepted_false_accepts, ["system_id", "model_id", "scenario_type", "judge_id", "coverage"])
    by_split.to_csv(out_dir / "false_accept_summary_by_split.csv", index=False)
    by_model = _false_accept_group_summary(accepted_false_accepts, ["system_id", "model_id", "judge_id", "coverage"])
    by_model.to_csv(out_dir / "false_accept_summary_by_model.csv", index=False)
    tag_counts = _tag_counts(accepted_false_accepts)
    tag_counts.to_csv(out_dir / "false_accept_tag_counts.csv", index=False)
    system_summary = pd.DataFrame(system_summaries)
    verdict = _forensics_verdict(accepted_false_accepts)
    summary = {
        "verdict": verdict,
        "false_accept_count": int(len(accepted_false_accepts)),
        "system_summary": system_summary.to_dict(orient="records"),
        "top_tags": tag_counts.head(10).to_dict(orient="records"),
        "worst_false_accepts": accepted_false_accepts.sort_values("rmse", ascending=False).head(10).to_dict(orient="records"),
    }
    _write_json(out_dir / "false_accept_forensics_summary.json", summary)
    write_false_accept_report(summary, system_summary, by_model, by_split, tag_counts, accepted_false_accepts, Path("reports/accepted_false_accept_forensics.md"))
    return summary


def _false_accept_group_summary(false_accepts: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if false_accepts.empty:
        return pd.DataFrame(columns=[*group_cols, "accepted_bad_count", "mean_bad_rmse", "median_bad_rmse", "worst_bad_rmse"])
    grouped = false_accepts.groupby(group_cols, as_index=False).agg(
        accepted_bad_count=("scenario_id", "count"),
        mean_bad_rmse=("rmse", "mean"),
        median_bad_rmse=("rmse", "median"),
        worst_bad_rmse=("rmse", "max"),
        mean_support_distance=("support_distance", "mean"),
        mean_uncertainty_score=("uncertainty_score", "mean"),
        mean_disagreement_score=("disagreement_score", "mean"),
        mean_invariant_residual=("invariant_residual", "mean"),
        mean_repair_amount=("repair_amount", "mean"),
    )
    return grouped


def _tag_counts(false_accepts: pd.DataFrame) -> pd.DataFrame:
    counts: dict[str, int] = {}
    for tag_string in false_accepts.get("tags", pd.Series(dtype=str)).astype(str):
        for tag in tag_string.split(";"):
            if tag:
                counts[tag] = counts.get(tag, 0) + 1
    return pd.DataFrame([{"tag": tag, "count": count} for tag, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))])


def _forensics_verdict(false_accepts: pd.DataFrame) -> str:
    if len(false_accepts) < 5:
        return "INCONCLUSIVE"
    model_share = float(false_accepts["model_id"].value_counts(normalize=True).iloc[0])
    scenario_share = float(false_accepts["scenario_type"].value_counts(normalize=True).iloc[0])
    if model_share >= 0.50 or scenario_share >= 0.50:
        return "FALSE_ACCEPTS_EXPLAINED"
    low_tags = [tag for tag in false_accepts["tags"].astype(str) if tag.count("LOW_") >= 4]
    if len(low_tags) / len(false_accepts) >= 0.30:
        return "FALSE_ACCEPTS_SIGNAL_BLIND_SPOT"
    return "INCONCLUSIVE"


def write_false_accept_report(
    summary: dict[str, Any],
    system_summary: pd.DataFrame,
    by_model: pd.DataFrame,
    by_split: pd.DataFrame,
    tag_counts: pd.DataFrame,
    false_accepts: pd.DataFrame,
    output: Path,
) -> None:
    _ensure_dir(output.parent)
    worst = false_accepts.sort_values("rmse", ascending=False).head(12).copy()
    text = f"""# Accepted False-Accept Forensics

## Question

Which bad scenarios are still accepted by the calibrated judge?

## False accepts by system

{_markdown_table(system_summary.rename(columns={"system_id": "system"}), ["system", "coverage", "accepted_count", "accepted_bad_count", "accepted_bad_rate"])}

## False accepts by model

{_markdown_table(by_model.rename(columns={"system_id": "system", "model_id": "model"}), ["system", "model", "accepted_bad_count", "mean_bad_rmse"], max_rows=40)}

## False accepts by scenario type

{_markdown_table(by_split.rename(columns={"system_id": "system"}), ["system", "scenario_type", "accepted_bad_count", "mean_bad_rmse"], max_rows=40)}

## Diagnostic tag counts

{_markdown_table(tag_counts, ["tag", "count"])}

## Worst accepted false accepts

{_markdown_table(worst.rename(columns={"system_id": "system", "model_id": "model"}), ["system", "scenario_id", "model", "scenario_type", "rmse", "tags"], max_rows=12)}

## Interpretation

The tags are diagnostic only and are defined in `docs/false_accept_forensics_tags.md`.

## Verdict

{summary["verdict"]}

## Recommended next action

Inspect the dominant model/scenario clusters before changing any refusal signal.
"""
    output.write_text(text, encoding="utf-8")


def load_event_config(path: str | Path) -> dict[str, Any]:
    config = _load_yaml(path)
    required = {"audit_id", "systems", "primary_coverages", "bad_label_modes"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"missing event audit config keys: {missing}")
    for system_id, system_config in config["systems"].items():
        if not system_config.get("thresholds"):
            raise ValueError(f"missing explicit event thresholds for {system_id}")
    return config


def _event_flags(system_id: str, states: np.ndarray, thresholds: dict[str, float]) -> dict[str, bool]:
    states = np.asarray(states, dtype=float)
    if system_id == "cstr":
        temp = bool(np.any(states[:, 1] > float(thresholds["temperature_high"])))
        conc = bool(
            np.any(states[:, 0] < float(thresholds["concentration_low"]))
            or np.any(states[:, 0] > float(thresholds["concentration_high"]))
        )
        return {
            "temperature_above_limit": temp,
            "concentration_out_of_safe_range": conc,
            "unsafe_reactor_state": bool(temp or conc),
        }
    if system_id == "two_tank":
        overflow = bool(np.any(states > float(thresholds["level_max"])))
        underflow = bool(np.any(states < float(thresholds["level_min"])))
        return {"overflow_event": overflow, "underflow_event": underflow}
    raise ValueError(f"unsupported event system: {system_id}")


def _load_system_experiment_config(system_id: str) -> dict[str, Any]:
    return _load_yaml(SYSTEM_CONFIGS[system_id])


def _event_label_table(system_id: str, event_config: dict[str, Any]) -> pd.DataFrame:
    experiment_config = _load_system_experiment_config(system_id)
    dataset = load_dataset(SYSTEM_RESULT_DIRS[system_id] / "data")
    models = [make_model(str(model_id), seed=int(experiment_config["seed"]) + idx) for idx, model_id in enumerate(experiment_config["models"])]
    for model in models:
        model.fit(dataset["model_train"])
    thresholds = event_config["systems"][system_id]["thresholds"]
    labels = event_config["systems"][system_id]["event_labels"]
    rows = []
    for split, batch in dataset.items():
        if not split.startswith("judge_test"):
            continue
        for i in range(batch.n_trajectories):
            scenario_id = f"{split}_{i:04d}"
            true_states = batch.states[i]
            true_flags = _event_flags(system_id, true_states, thresholds)
            for model in models:
                predicted = model.predict_rollout(batch.states[i, 0], batch.actions[i], batch.disturbances[i])
                pred_flags = _event_flags(system_id, predicted, thresholds)
                row = {
                    "system_id": system_id,
                    "scenario_id": scenario_id,
                    "model_id": model.model_id,
                    "split": split,
                    "scenario_type": batch.scenario_type[i],
                    "true_any_event": bool(any(true_flags[label] for label in labels)),
                    "predicted_any_event": bool(any(pred_flags[label] for label in labels)),
                }
                for label in labels:
                    row[f"true_{label}"] = true_flags[label]
                    row[f"predicted_{label}"] = pred_flags[label]
                    row[f"bad_{label}"] = bool(true_flags[label] != pred_flags[label])
                row["bad_event"] = bool(any(row[f"bad_{label}"] for label in labels))
                rows.append(row)
    return pd.DataFrame(rows)


def _precision_recall(predicted: pd.Series, actual: pd.Series) -> tuple[float, float]:
    pred = predicted.astype(bool).to_numpy()
    act = actual.astype(bool).to_numpy()
    tp = float(np.sum(pred & act))
    fp = float(np.sum(pred & ~act))
    fn = float(np.sum(~pred & act))
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    return precision, recall


def _event_risk_rows(merged: pd.DataFrame, judges: list[str], coverages: list[float], modes: list[str]) -> pd.DataFrame:
    rows = []
    merged = merged.copy()
    merged["bad_rmse"] = merged["bad_rmse_label"].astype(bool)
    merged["bad_rmse_or_event"] = merged["bad_rmse"] | merged["bad_event"].astype(bool)
    for system_id, system_frame in merged.groupby("system_id", sort=True):
        for model_id, model_frame in system_frame.groupby("model_id", sort=True):
            precision, recall = _precision_recall(model_frame["predicted_any_event"], model_frame["true_any_event"])
            event_bad_rate = float(model_frame["bad_event"].mean())
            for judge in judges:
                for coverage in coverages:
                    accepted_frames = []
                    for _, group in model_frame.groupby("scenario_type", sort=False):
                        accepted_frames.append(group.loc[_accepted_mask_for_group(group, judge, coverage)])
                    accepted = pd.concat(accepted_frames, ignore_index=True) if accepted_frames else pd.DataFrame()
                    accepted_count = len(accepted)
                    for mode in modes:
                        accepted_bad_count = int(accepted[mode].astype(bool).sum()) if accepted_count else 0
                        accepted_event_bad_count = int(accepted["bad_event"].astype(bool).sum()) if accepted_count else 0
                        rows.append(
                            {
                                "system_id": system_id,
                                "model_id": model_id,
                                "judge_id": judge,
                                "coverage": float(coverage),
                                "bad_label_mode": mode,
                                "false_accept_rate": float(accepted_bad_count / accepted_count) if accepted_count else 0.0,
                                "event_false_accept_rate": float(accepted_event_bad_count / accepted_count) if accepted_count else 0.0,
                                "accepted_count": int(accepted_count),
                                "accepted_bad_count": accepted_bad_count,
                                "accepted_event_bad_count": accepted_event_bad_count,
                                "event_precision": precision,
                                "event_recall": recall,
                                "event_bad_rate": event_bad_rate,
                            }
                        )
    return pd.DataFrame(rows)


def analyze_event_risk(
    effect_config_path: str | Path,
    event_config_path: str | Path,
    output: str | Path,
    report_path: str | Path = "reports/event_risk_audit.md",
) -> dict[str, Any]:
    effect_config = load_effect_audit_config(effect_config_path)
    event_config = load_event_config(event_config_path)
    out_dir = Path(output)
    _ensure_dir(out_dir)
    all_labels = []
    availability = []
    for system_id in effect_config["systems"]:
        try:
            labels = _event_label_table(system_id, event_config)
            test_table = pd.read_csv(SYSTEM_RESULT_DIRS[system_id] / "test_table.csv")
            merged = test_table.merge(labels, on=["system_id", "scenario_id", "model_id", "split", "scenario_type"], how="left")
            if merged["bad_event"].isna().any():
                raise RuntimeError(f"missing event labels for {system_id}")
            availability.append({"system": system_id, "event_labels_available": True, "missing_reason": "none"})
            all_labels.append(merged)
        except Exception as exc:
            availability.append({"system": system_id, "event_labels_available": False, "missing_reason": str(exc)})
    event_labels = pd.concat(all_labels, ignore_index=True) if all_labels else pd.DataFrame()
    event_labels.to_csv(out_dir / "event_labels.csv", index=False)
    if any(not item["event_labels_available"] for item in availability):
        empty_risk = pd.DataFrame(
            columns=[
                "system_id",
                "model_id",
                "judge_id",
                "coverage",
                "bad_label_mode",
                "false_accept_rate",
                "event_false_accept_rate",
                "accepted_count",
                "accepted_bad_count",
                "accepted_event_bad_count",
                "event_precision",
                "event_recall",
                "event_bad_rate",
            ]
        )
        empty_risk.to_csv(out_dir / "event_risk_by_judge.csv", index=False)
        summary = {
            "verdict": "EVENT_UNAVAILABLE",
            "event_label_availability": availability,
            "comparison_rows": [],
            "event_thresholds": event_config["systems"],
        }
        _write_json(out_dir / "event_risk_summary.json", summary)
        write_event_risk_report(event_config, summary, pd.DataFrame(), Path(report_path))
        return summary
    judges = [str(effect_config["baseline_judge"]), *[str(value) for value in effect_config["calibrated_family"]]]
    risk = _event_risk_rows(
        event_labels,
        judges=judges,
        coverages=[float(value) for value in event_config["primary_coverages"]],
        modes=[str(value) for value in event_config["bad_label_modes"]],
    )
    risk.to_csv(out_dir / "event_risk_by_judge.csv", index=False)
    summary_rows = _event_summary_rows(effect_config, risk)
    verdict = _event_verdict(summary_rows)
    summary = {
        "verdict": verdict,
        "event_label_availability": availability,
        "comparison_rows": summary_rows.to_dict(orient="records"),
        "event_thresholds": event_config["systems"],
    }
    _write_json(out_dir / "event_risk_summary.json", summary)
    write_event_risk_report(event_config, summary, summary_rows, Path(report_path))
    return summary


def _event_summary_rows(effect_config: dict[str, Any], risk: pd.DataFrame) -> pd.DataFrame:
    rows = []
    baseline = str(effect_config["baseline_judge"])
    for system_id in effect_config["systems"]:
        for coverage in [float(value) for value in effect_config["primary_coverages"]]:
            calibrated_judge = _best_calibrated_judge(system_id, coverage)
            for judge in [baseline, calibrated_judge]:
                frame = risk[
                    (risk["system_id"] == system_id)
                    & np.isclose(risk["coverage"], coverage)
                    & (risk["judge_id"] == judge)
                ]
                pivot = frame.groupby("bad_label_mode")["false_accept_rate"].mean()
                event_counts = frame.groupby("bad_label_mode")["accepted_event_bad_count"].sum()
                rows.append(
                    {
                        "system_id": system_id,
                        "coverage": coverage,
                        "judge": judge,
                        "rmse_far": float(pivot.get("bad_rmse", 0.0)),
                        "event_far": float(pivot.get("bad_event", 0.0)),
                        "rmse_or_event_far": float(pivot.get("bad_rmse_or_event", 0.0)),
                        "accepted_event_bad_count": int(event_counts.get("bad_event", 0)),
                    }
                )
    return pd.DataFrame(rows)


def _event_verdict(summary_rows: pd.DataFrame) -> str:
    support = []
    weaken = []
    for system_id, system_frame in summary_rows.groupby("system_id", sort=True):
        system_support = False
        system_weaken = False
        for coverage, coverage_frame in system_frame.groupby("coverage", sort=True):
            baseline = coverage_frame[coverage_frame["judge"] == "best_single_signal_selected_on_calibration"]
            calibrated = coverage_frame[coverage_frame["judge"] != "best_single_signal_selected_on_calibration"]
            if baseline.empty or calibrated.empty:
                continue
            event_margin = float(baseline["event_far"].iloc[0] - calibrated["event_far"].iloc[0])
            if event_margin > 1e-12 or float(calibrated["event_far"].iloc[0]) <= 0.01:
                system_support = True
            if event_margin <= 1e-12 and int(calibrated["accepted_event_bad_count"].iloc[0]) > 0:
                system_weaken = True
        support.append(system_support)
        weaken.append(system_weaken)
    if all(support):
        return "EVENT_SUPPORTS_CLAIM"
    if any(weaken):
        return "EVENT_WEAKENS_CLAIM"
    if any(support):
        return "MIXED_EVENT_EVIDENCE"
    return "EVENT_UNAVAILABLE"


def write_event_risk_report(event_config: dict[str, Any], summary: dict[str, Any], rows: pd.DataFrame, output: Path) -> None:
    _ensure_dir(output.parent)
    availability = pd.DataFrame(summary["event_label_availability"])
    text = f"""# Event-Risk Audit

## Question

Does calibrated low-coverage refusal reduce event-risk false accepts, not only RMSE false accepts?

## Event definitions

{json.dumps(event_config["systems"], indent=2, sort_keys=True)}

## Event-label availability

{_markdown_table(availability, ["system", "event_labels_available", "missing_reason"])}

## Bad RMSE vs bad event comparison

{_markdown_table(rows.rename(columns={"system_id": "system"}), ["system", "coverage", "judge", "rmse_far", "event_far", "rmse_or_event_far"])}

## CSTR event-risk result

{_system_event_text(rows, "cstr")}

## TwoTank event-risk result

{_system_event_text(rows, "two_tank")}

## Verdict

{summary["verdict"]}

## Explanation

Event labels were computed from true and predicted trajectories using explicit thresholds, then merged with deployable judge risk rankings.

## Known limitations

Predicted trajectories are regenerated from saved model-train/test data because earlier calibrated artifacts did not store every prediction trajectory.
"""
    output.write_text(text, encoding="utf-8")


def _system_event_text(rows: pd.DataFrame, system_id: str) -> str:
    if rows.empty or "system_id" not in rows.columns:
        return "event labels unavailable"
    subset = rows[rows["system_id"] == system_id]
    return "\n".join(
        f"- coverage {row.coverage:g}, judge {row.judge}: event FAR {row.event_far:.6f}, RMSE-or-event FAR {row.rmse_or_event_far:.6f}"
        for row in subset.itertuples()
    )


def make_practical_utility_decision_gate(
    effect_size: str | Path,
    forensics: str | Path,
    event_risk: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    effect = _load_json(effect_size)
    forensic = _load_json(forensics)
    event = _load_json(event_risk)
    effect_verdict = effect["verdict"]
    forensic_verdict = forensic["verdict"]
    event_verdict = event["verdict"]
    if (
        effect_verdict == "STRONG_TWO_SYSTEM_EFFECT"
        and event_verdict in {"EVENT_SUPPORTS_CLAIM", "MIXED_EVENT_EVIDENCE"}
        and forensic_verdict != "FALSE_ACCEPTS_SIGNAL_BLIND_SPOT"
    ):
        decision = "WRITE_TECHNICAL_REPORT"
    elif effect_verdict == "WEAK_TWO_SYSTEM_EFFECT" and event_verdict != "EVENT_WEAKENS_CLAIM":
        decision = "NARROW_TO_WEAK_LOW_COVERAGE_CLAIM"
    elif event_verdict == "EVENT_WEAKENS_CLAIM" and effect_verdict in {"WEAK_TWO_SYSTEM_EFFECT", "MIXED_EFFECT"}:
        decision = "REDESIGN_TARGET"
    elif effect_verdict in {"STRONG_TWO_SYSTEM_EFFECT", "WEAK_TWO_SYSTEM_EFFECT", "MIXED_EFFECT"} and forensic_verdict != "FALSE_ACCEPTS_SIGNAL_BLIND_SPOT":
        decision = "ADD_THIRD_SYSTEM_ONLY_AFTER_FIXES"
    else:
        decision = "DO_NOT_EXPAND"
    allowed_claims = {
        "WRITE_TECHNICAL_REPORT": "A bounded technical report on calibrated low-coverage refusal across TwoTank and CSTR.",
        "NARROW_TO_WEAK_LOW_COVERAGE_CLAIM": "A weak but positive low-coverage result under the frozen protocol.",
        "REDESIGN_TARGET": "RMSE refusal is not the right target; event-risk target design should be reconsidered.",
        "ADD_THIRD_SYSTEM_ONLY_AFTER_FIXES": "No third-system expansion until the named weakness is fixed and frozen.",
        "DO_NOT_EXPAND": "No expansion or broad claim is supported by this audit.",
    }
    expansion_allowed = decision == "ADD_THIRD_SYSTEM_ONLY_AFTER_FIXES"
    result = {
        "effect_size_verdict": effect_verdict,
        "false_accept_forensics_verdict": forensic_verdict,
        "event_risk_verdict": event_verdict,
        "decision": decision,
        "allowed_claim": allowed_claims[decision],
        "forbidden_claims": [
            "safety certification",
            "product readiness",
            "plant-wide generalization",
            "autonomous control",
            "universal simulator reliability",
        ],
        "recommended_next_action": _recommended_action(decision),
        "expansion_allowed": expansion_allowed,
    }
    output_path = Path(output)
    _ensure_dir(output_path.parent)
    _write_json(output_path.with_suffix(".json"), result)
    write_practical_gate_report(result, output_path)
    return result


def _recommended_action(decision: str) -> str:
    if decision == "WRITE_TECHNICAL_REPORT":
        return "Draft a bounded technical report with explicit low-coverage and event-risk limitations."
    if decision == "NARROW_TO_WEAK_LOW_COVERAGE_CLAIM":
        return "Use weaker wording and diagnose the small CSTR margin before expansion."
    if decision == "REDESIGN_TARGET":
        return "Define a pre-registered event-risk target and rerun from protocol lock."
    if decision == "ADD_THIRD_SYSTEM_ONLY_AFTER_FIXES":
        return "Fix the identified weakness before any third-system replication."
    return "Do not expand; redesign or downgrade the claim."


def write_practical_gate_report(result: dict[str, Any], output: Path) -> None:
    _ensure_dir(output.parent)
    text = f"""# Practical Utility Decision Gate

## Starting point

The frozen-protocol calibrated judge passed on TwoTank and CSTR, but CSTR margin was small.

## Effect-size verdict

{result["effect_size_verdict"]}

## False-accept forensics verdict

{result["false_accept_forensics_verdict"]}

## Event-risk verdict

{result["event_risk_verdict"]}

## Practical threshold check

See `results/effect_size_audit/effect_size/effect_size_by_system.csv`.

## Decision

{result["decision"]}

## Allowed claims

{result["allowed_claim"]}

## Forbidden claims

{", ".join(result["forbidden_claims"])}

## Recommended next action

{result["recommended_next_action"]}

## Explanation

Decision follows the fixed practical utility gate rules. No benchmark expansion is authorized unless explicitly stated by the decision.
"""
    output.write_text(text, encoding="utf-8")
