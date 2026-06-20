from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SIMPLE_JUDGES = [
    "support_only",
    "uncertainty_only",
    "disagreement_only",
    "invariant_only",
    "repair_only",
    "random_baseline",
]
ORACLE_JUDGE = "oracle_error_rank"
COMBINED_JUDGE = "combined_linear"


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
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _verdict(overall_win_rate: float, ood_split_win_count: int, winning_best_simple: pd.Series) -> str:
    only_beats_random = not winning_best_simple.empty and set(winning_best_simple) == {"random_baseline"}
    if overall_win_rate < 0.40 or only_beats_random:
        return "NOT_SUPPORTED"
    if overall_win_rate >= 0.70 and ood_split_win_count >= 2:
        return "SUPPORTED"
    return "MIXED"


def compute_claim_audit(results_dir: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = Path(results_dir)
    risk = pd.read_csv(source / "risk_coverage.csv")
    required = {"split", "model_id", "judge_id", "coverage", "false_accept_rate"}
    missing = sorted(required - set(risk.columns))
    if missing:
        raise ValueError(f"risk_coverage.csv missing columns required for claim audit: {missing}")
    if not np.isfinite(risk.select_dtypes(include=[float, int]).to_numpy()).all():
        raise ValueError("risk_coverage.csv contains non-finite values")

    rows = []
    for (split, model_id, coverage), group in risk.groupby(["split", "model_id", "coverage"], sort=False):
        by_judge = group.set_index("judge_id")["false_accept_rate"]
        missing_judges = [judge for judge in [COMBINED_JUDGE, *SIMPLE_JUDGES] if judge not in by_judge.index]
        if missing_judges:
            raise ValueError(f"missing judges for {split}/{model_id}/{coverage}: {missing_judges}")
        simple_scores = by_judge[SIMPLE_JUDGES].sort_values(kind="mergesort")
        best_simple_judge = str(simple_scores.index[0])
        best_simple_far = float(simple_scores.iloc[0])
        combined_far = float(by_judge[COMBINED_JUDGE])
        margin = best_simple_far - combined_far
        rows.append(
            {
                "split": split,
                "model_id": model_id,
                "coverage": float(coverage),
                "combined_far": combined_far,
                "best_simple_judge": best_simple_judge,
                "best_simple_far": best_simple_far,
                "combined_margin": float(margin),
                "combined_wins": bool(combined_far < best_simple_far),
                "combined_ties": bool(np.isclose(combined_far, best_simple_far)),
            }
        )

    audit = pd.DataFrame(rows).sort_values(["split", "model_id", "coverage"]).reset_index(drop=True)
    split_rates = audit.groupby("split", as_index=False)["combined_wins"].mean().rename(columns={"combined_wins": "combined_win_rate"})
    model_rates = audit.groupby("model_id", as_index=False)["combined_wins"].mean().rename(columns={"combined_wins": "combined_win_rate"})
    coverage_rates = audit.groupby("coverage", as_index=False)["combined_wins"].mean().rename(columns={"combined_wins": "combined_win_rate"})
    overall_win_rate = float(audit["combined_wins"].mean())
    ood_split_rates = split_rates[split_rates["split"] != "id_test"]
    ood_split_win_count = int((ood_split_rates["combined_win_rate"] > 0.5).sum())
    winning_best_simple = audit.loc[audit["combined_wins"], "best_simple_judge"]
    verdict = _verdict(overall_win_rate, ood_split_win_count, winning_best_simple)

    real_judges = [COMBINED_JUDGE, *SIMPLE_JUDGES]
    oracle_gap_rows = []
    if ORACLE_JUDGE in risk["judge_id"].unique():
        for (split, model_id, coverage), group in risk.groupby(["split", "model_id", "coverage"], sort=False):
            by_judge = group.set_index("judge_id")["false_accept_rate"]
            if ORACLE_JUDGE in by_judge.index:
                best_real_far = float(by_judge[real_judges].min())
                oracle_far = float(by_judge[ORACLE_JUDGE])
                oracle_gap_rows.append(
                    {
                        "split": split,
                        "model_id": model_id,
                        "coverage": float(coverage),
                        "best_real_far": best_real_far,
                        "oracle_far": oracle_far,
                        "oracle_gap": best_real_far - oracle_far,
                    }
                )
    oracle_gap = pd.DataFrame(oracle_gap_rows)

    best_simple_by_model = (
        audit.groupby(["model_id", "best_simple_judge"]).size().reset_index(name="count")
        .sort_values(["model_id", "count"], ascending=[True, False])
        .drop_duplicates("model_id")
    )
    model_report = model_rates.merge(best_simple_by_model[["model_id", "best_simple_judge"]], on="model_id", how="left")
    model_report["verdict"] = np.where(model_report["combined_win_rate"] >= 0.7, "SUPPORTED", np.where(model_report["combined_win_rate"] >= 0.4, "MIXED", "NOT_SUPPORTED"))

    summary = {
        "results_dir": str(source),
        "verdict": verdict,
        "overall_win_rate": overall_win_rate,
        "win_rate_by_split": split_rates.to_dict(orient="records"),
        "win_rate_by_model": model_rates.to_dict(orient="records"),
        "win_rate_by_coverage": coverage_rates.to_dict(orient="records"),
        "ood_split_win_count": ood_split_win_count,
        "best_simple_judge_overall": str(audit["best_simple_judge"].mode().iloc[0]),
        "oracle_gap_mean": float(oracle_gap["oracle_gap"].mean()) if not oracle_gap.empty else None,
        "oracle_gap_min": float(oracle_gap["oracle_gap"].min()) if not oracle_gap.empty else None,
        "oracle_gap_max": float(oracle_gap["oracle_gap"].max()) if not oracle_gap.empty else None,
        "model_report": model_report.to_dict(orient="records"),
    }
    return audit, summary


def write_claim_audit(results_dir: str | Path, report_path: str | Path = "reports/v0_claim_audit.md") -> dict[str, Any]:
    source = Path(results_dir)
    audit, summary = compute_claim_audit(source)
    source.mkdir(parents=True, exist_ok=True)
    audit_path = source / "claim_audit.csv"
    json_path = source / "claim_audit.json"
    audit.to_csv(audit_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    split_table = (
        audit[audit["split"] != "id_test"]
        .groupby(["split", "coverage", "best_simple_judge"], as_index=False)
        .agg(
            best_simple_far=("best_simple_far", "mean"),
            combined_far=("combined_far", "mean"),
            combined_margin=("combined_margin", "mean"),
            combined_wins=("combined_wins", "mean"),
        )
        .sort_values(["split", "coverage"])
    )
    model_table = pd.DataFrame(summary["model_report"])
    coverage_table = pd.DataFrame(summary["win_rate_by_coverage"])
    oracle_gap_text = (
        "Oracle gap was not available."
        if summary["oracle_gap_mean"] is None
        else (
            f"Mean oracle gap: {summary['oracle_gap_mean']:.6f}; "
            f"min: {summary['oracle_gap_min']:.6f}; max: {summary['oracle_gap_max']:.6f}."
        )
    )
    explanation = (
        f"Overall strict win rate was {summary['overall_win_rate']:.6f}. "
        f"Combined_linear won on {summary['ood_split_win_count']} OOD splits under the split-level rule. "
        f"Verdict: {summary['verdict']}."
    )
    text = f"""# v0 Claim Audit

## Main question

Did combined_linear reduce false_accept_rate compared with the strongest simple judge?

## Data sources

- {source / "risk_coverage.csv"}
- {source / "scenario_scores.csv"}
- {source / "model_metrics.csv"}
- {source / "summary.json"}

## Judge definitions

Real judges: {", ".join([COMBINED_JUDGE, *SIMPLE_JUDGES])}.
Diagnostic judge excluded from real-method ranking: {ORACLE_JUDGE}.

## Result by OOD split

{_markdown_table(split_table, ["split", "coverage", "best_simple_judge", "best_simple_far", "combined_far", "combined_margin", "combined_wins"])}

## Result by model

{_markdown_table(model_table, ["model_id", "best_simple_judge", "combined_win_rate", "verdict"])}

## Result by coverage

{_markdown_table(coverage_table, ["coverage", "combined_win_rate"])}

## Oracle gap

{oracle_gap_text}

## Verdict

{summary["verdict"]}

## Explanation

{explanation}

## Known failure modes

- Ties are not counted as wins.
- Oracle_error_rank is diagnostic only and excluded from strongest-simple comparisons.
"""
    report_target = Path(report_path)
    report_target.parent.mkdir(parents=True, exist_ok=True)
    report_target.write_text(text, encoding="utf-8")
    return summary

