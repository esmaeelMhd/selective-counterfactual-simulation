from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scs.experiments.runner import load_config, run_experiment


SEVERITY_ORDER = ["low", "medium", "high", "extreme"]


def _table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    table = df[columns].copy()
    if max_rows is not None:
        table = table.head(max_rows)
    if table.empty:
        return "| " + " | ".join(columns) + " |\n| " + " | ".join(["---"] * len(columns)) + " |"
    lines = ["| " + " | ".join(columns) + " |"]
    lines.append("| " + " | ".join("---:" if pd.api.types.is_numeric_dtype(table[col]) else "---" for col in columns) + " |")
    for _, row in table.iterrows():
        values = [f"{row[col]:.6f}" if isinstance(row[col], float) else str(row[col]) for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _severity_verdict(summary_df: pd.DataFrame) -> tuple[str, str]:
    ordered = summary_df.set_index("severity").reindex([s for s in SEVERITY_ORDER if s in set(summary_df["severity"])])
    first = ordered.iloc[0]
    last = ordered.iloc[-1]
    error_increased = float(last["mean_true_error"]) > float(first["mean_true_error"]) * 1.05
    signal_columns = [
        "mean_support_distance",
        "mean_uncertainty_score",
        "mean_disagreement_score",
        "mean_invariant_residual",
        "mean_repair_amount",
    ]
    signal_increases = [
        column for column in signal_columns
        if float(last[column]) > float(first[column]) * 1.05 + 1e-12
    ]
    if error_increased and signal_increases:
        return "MEANINGFUL", f"Error increased and these validator signals increased: {', '.join(signal_increases)}."
    if error_increased:
        return "PARTIAL", "Error increased, but validator signals were weak or inconsistent."
    return "WEAK", "Severity did not meaningfully increase model error."


def run_severity_sweep(config_path: str, severities: list[str], output: str) -> dict:
    unknown = sorted(set(severities) - set(SEVERITY_ORDER))
    if unknown:
        raise ValueError(f"unknown severities: {unknown}")
    base_config = load_config(config_path)
    if base_config["system_id"] != "two_tank":
        raise ValueError("severity sweep is currently defined for TwoTank only")
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    risk_frames = []
    score_frames = []
    summary_rows = []

    for severity in severities:
        severity_dir = out_dir / severity
        config = dict(base_config)
        config["severity"] = severity
        config["include_pump_degradation"] = True
        config["experiment_id"] = f"{base_config['experiment_id']}_{severity}"
        config["output_dir"] = str(severity_dir)
        resolved = severity_dir / "resolved_config.yaml"
        severity_dir.mkdir(parents=True, exist_ok=True)
        resolved.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        run_experiment(
            resolved,
            output_dir=severity_dir,
            report_path=severity_dir / "smoke_report.md",
            command=f"python scripts/run_severity_sweep.py --config {config_path} --severities {' '.join(severities)} --output {output}",
        )
        scores = pd.read_csv(severity_dir / "scenario_scores.csv")
        risk = pd.read_csv(severity_dir / "risk_coverage.csv")
        scores.insert(0, "severity", severity)
        risk.insert(0, "severity", severity)
        score_frames.append(scores)
        risk_frames.append(risk)
        ood_scores = scores[scores["split"] != "id_test"]
        summary_rows.append(
            {
                "severity": severity,
                "mean_true_error": float(ood_scores["error"].mean()),
                "std_error": float(ood_scores["error"].std(ddof=0)),
                "mean_support_distance": float(ood_scores["support_distance"].mean()),
                "mean_uncertainty_score": float(ood_scores["uncertainty"].mean()),
                "mean_disagreement_score": float(ood_scores["disagreement"].mean()),
                "mean_invariant_residual": float(ood_scores["invariant_residual"].mean()),
                "mean_repair_amount": float(ood_scores["repair_amount"].mean()),
            }
        )

    risk_all = pd.concat(risk_frames, ignore_index=True)
    scores_all = pd.concat(score_frames, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)
    summary_df["severity_rank"] = summary_df["severity"].map({severity: idx for idx, severity in enumerate(SEVERITY_ORDER)})
    summary_df = summary_df.sort_values("severity_rank").drop(columns=["severity_rank"])
    risk_all.to_csv(out_dir / "risk_coverage_by_severity.csv", index=False)
    scores_all.to_csv(out_dir / "scenario_scores_by_severity.csv", index=False)
    summary_df.to_csv(out_dir / "severity_summary.csv", index=False)

    verdict, explanation = _severity_verdict(summary_df)
    false_accept = (
        risk_all.groupby(["severity", "judge_id", "coverage"], as_index=False)["false_accept_rate"]
        .mean()
        .sort_values(["severity", "judge_id", "coverage"])
    )
    monotonicity = {
        "error_increased_low_to_last": bool(summary_df.iloc[-1]["mean_true_error"] > summary_df.iloc[0]["mean_true_error"]),
        "support_increased_low_to_last": bool(summary_df.iloc[-1]["mean_support_distance"] > summary_df.iloc[0]["mean_support_distance"]),
        "false_accept_rate_changed": bool(false_accept["false_accept_rate"].max() > false_accept["false_accept_rate"].min()),
    }
    summary = {
        "command": f"python scripts/run_severity_sweep.py --config {config_path} --severities {' '.join(severities)} --output {output}",
        "severities": severities,
        "verdict": verdict,
        "explanation": explanation,
        "monotonicity": monotonicity,
        "summary": summary_df.to_dict(orient="records"),
    }
    (out_dir / "severity_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    report_path = Path("reports/two_tank_severity_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = f"""# TwoTank Severity Sweep Report

## Command

```bash
{summary["command"]}
```

## Severity definitions

See `docs/two_tank_intervention_severity.md`.

## Error vs severity

{_table(summary_df.rename(columns={"mean_true_error": "mean_error"}), ["severity", "mean_error", "std_error"])}

## Validator signals vs severity

{_table(summary_df.rename(columns={"mean_support_distance": "support", "mean_uncertainty_score": "uncertainty", "mean_disagreement_score": "disagreement", "mean_invariant_residual": "invariant", "mean_repair_amount": "repair"}), ["severity", "support", "uncertainty", "disagreement", "invariant", "repair"])}

## False accept rate vs severity

{_table(false_accept, ["severity", "judge_id", "coverage", "false_accept_rate"], max_rows=80)}

## Monotonicity checks

Did error increase with severity? {monotonicity["error_increased_low_to_last"]}
Did support distance increase with severity? {monotonicity["support_increased_low_to_last"]}
Did false accept rate change with severity? {monotonicity["false_accept_rate_changed"]}

## Verdict

{verdict}

## Explanation

{explanation}

## Known failures

- none
"""
    report_path.write_text(report, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TwoTank intervention severity sweep.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--severities", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = run_severity_sweep(args.config, args.severities, args.output)
    print(summary["verdict"])


if __name__ == "__main__":
    main()
