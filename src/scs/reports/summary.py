from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_summary_json(summary: dict, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def _table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "empty"
    return df.head(max_rows).to_string(index=False)


def _dataset_summary_frame(summary: dict) -> pd.DataFrame:
    dataset_summary = summary["dataset_summary"]
    first_value = next(iter(dataset_summary.values())) if dataset_summary else {}
    if isinstance(first_value, dict) and "train" in first_value:
        rows = []
        for system_id, splits in dataset_summary.items():
            for split, values in splits.items():
                rows.append({"system": system_id, "split": split, **values})
        return pd.DataFrame(rows)
    return pd.DataFrame.from_dict(dataset_summary, orient="index").reset_index().rename(columns={"index": "split"})


def write_smoke_report(
    summary: dict,
    risk_coverage: pd.DataFrame,
    model_metrics: pd.DataFrame,
    output_path: str | Path,
    command: str,
) -> None:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    dataset_summary = _dataset_summary_frame(summary)
    id_perf = model_metrics[model_metrics["split"] == "id_test"]
    ood_perf = model_metrics[model_metrics["split"] != "id_test"]
    risk_summary = (
        risk_coverage.groupby(["judge_id", "coverage"], as_index=False)["false_accept_rate"]
        .mean()
        .sort_values(["coverage", "false_accept_rate", "judge_id"])
    )
    best_by_coverage = risk_summary.loc[
        risk_summary.groupby("coverage")["false_accept_rate"].idxmin()
    ].sort_values("coverage")

    known_failures = summary.get("known_failures", [])
    known_failure_text = "\n".join(f"- {item}" for item in known_failures) if known_failures else "- none"
    claim_status = summary.get(
        "claim_status",
        {
            "result": "NOT EVALUATED",
            "explanation": "This run did not include the v1 multi-system claim-status evaluation.",
        },
    )

    text = f"""# Smoke Report

## Experiment config

```json
{json.dumps(summary["config"], indent=2, sort_keys=True)}
```

## Dataset summary

```text
{_table(dataset_summary)}
```

## Model summary

```text
{_table(model_metrics[["model_id", "split", "rmse_mean", "mae_mean", "max_abs_error_mean"]])}
```

## In-distribution performance

```text
{_table(id_perf[["model_id", "rmse_mean", "mae_mean", "final_state_error_mean"]])}
```

## OOD performance

```text
{_table(ood_perf[["model_id", "split", "rmse_mean", "mae_mean", "final_state_error_mean"]])}
```

## Risk-coverage summary

risk_coverage_rows: {len(risk_coverage)}
scenario_score_rows: {summary.get("scenario_score_rows", "unknown")}

```text
{_table(risk_summary)}
```

## Best judge by coverage

```text
{_table(best_by_coverage)}
```

## Did combined judge beat simple judges?

{summary["combined_judge_result"]["statement"]}

## Claim status

Result: {claim_status["result"]}

Explanation:
{claim_status["explanation"]}

## Known failures

{known_failure_text}

## Reproduction command

```bash
{command}
```
"""
    target.write_text(text, encoding="utf-8")
