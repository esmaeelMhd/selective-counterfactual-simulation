from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

import pandas as pd
import yaml

from scs.experiments.runner import load_config, run_experiment
from scs.reports.claim_audit import write_claim_audit


def _table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "| " + " | ".join(columns) + " |\n| " + " | ".join(["---"] * len(columns)) + " |"
    lines = ["| " + " | ".join(columns) + " |"]
    lines.append("| " + " | ".join("---:" if pd.api.types.is_numeric_dtype(df[col]) else "---" for col in columns) + " |")
    for _, row in df[columns].iterrows():
        values = [f"{row[col]:.6f}" if isinstance(row[col], float) else str(row[col]) for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _seed_verdict(overall_rate: float, winning_seed_count: int, n_seeds: int) -> str:
    robust_seed_count = 7 if n_seeds >= 10 else max(1, int(0.7 * n_seeds))
    unstable_low = 4 if n_seeds >= 10 else max(1, int(0.4 * n_seeds))
    unstable_high = 6 if n_seeds >= 10 else max(1, int(0.6 * n_seeds))
    if overall_rate >= 0.70 and winning_seed_count >= robust_seed_count:
        return "ROBUST"
    if 0.40 <= overall_rate <= 0.70 or unstable_low <= winning_seed_count <= unstable_high:
        return "UNSTABLE"
    return "NOT_SUPPORTED"


def run_seed_sweep(config_path: str, seeds: list[int], output: str) -> dict:
    base_config = load_config(config_path)
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    risk_frames = []
    audit_rows = []
    failures = []

    for seed in seeds:
        seed_dir = out_dir / f"seed_{seed}"
        seed_config = dict(base_config)
        seed_config["seed"] = int(seed)
        seed_config["experiment_id"] = f"{base_config['experiment_id']}_seed_{seed}"
        seed_config["output_dir"] = str(seed_dir)
        resolved_config = seed_dir / "resolved_config.yaml"
        seed_dir.mkdir(parents=True, exist_ok=True)
        resolved_config.write_text(yaml.safe_dump(seed_config, sort_keys=False), encoding="utf-8")
        try:
            run_experiment(
                resolved_config,
                output_dir=seed_dir,
                report_path=seed_dir / "smoke_report.md",
                command=f"python scripts/run_seed_sweep.py --config {config_path} --seeds {' '.join(map(str, seeds))} --output {output}",
            )
            audit = write_claim_audit(seed_dir, seed_dir / "claim_audit.md")
            risk = pd.read_csv(seed_dir / "risk_coverage.csv")
            risk.insert(0, "seed", seed)
            risk_frames.append(risk)
            audit_rows.append(
                {
                    "seed": seed,
                    "status": "ok",
                    "verdict": audit["verdict"],
                    "overall_combined_win_rate": audit["overall_win_rate"],
                    "best_simple_judge": audit["best_simple_judge_overall"],
                    "notes": "",
                }
            )
        except Exception as exc:  # pragma: no cover - exercised by integration failure cases.
            failures.append({"seed": seed, "error": str(exc)})
            audit_rows.append(
                {
                    "seed": seed,
                    "status": "failed",
                    "verdict": "FAILED",
                    "overall_combined_win_rate": 0.0,
                    "best_simple_judge": "",
                    "notes": str(exc),
                }
            )

    if failures:
        (out_dir / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
        raise RuntimeError(f"seed sweep failed for {len(failures)} seeds; see {out_dir / 'failures.json'}")

    risk_all = pd.concat(risk_frames, ignore_index=True)
    audit_by_seed = pd.DataFrame(audit_rows)
    risk_all.to_csv(out_dir / "risk_coverage_all.csv", index=False)
    audit_by_seed.to_csv(out_dir / "claim_audit_by_seed.csv", index=False)

    rates = audit_by_seed["overall_combined_win_rate"].astype(float).tolist()
    winning_seed_count = int((audit_by_seed["overall_combined_win_rate"] >= 0.70).sum())
    overall_rate = float(mean(rates)) if rates else 0.0
    aggregate = {
        "overall_combined_win_rate": {
            "mean": overall_rate,
            "std": float(pstdev(rates)) if len(rates) > 1 else 0.0,
            "min": float(min(rates)) if rates else 0.0,
            "max": float(max(rates)) if rates else 0.0,
        }
    }
    split = []
    coverage = []
    claim_audits = []
    for seed in seeds:
        audit_path = out_dir / f"seed_{seed}" / "claim_audit.csv"
        seed_audit = pd.read_csv(audit_path)
        seed_audit.insert(0, "seed", seed)
        claim_audits.append(seed_audit)
    claim_all = pd.concat(claim_audits, ignore_index=True)
    split_df = claim_all.groupby(["seed", "split"], as_index=False)["combined_wins"].mean()
    split = (
        split_df.groupby("split", as_index=False)["combined_wins"]
        .agg(combined_win_rate_mean="mean", combined_win_rate_std="std")
        .fillna(0.0)
        .to_dict(orient="records")
    )
    coverage_df = claim_all.groupby(["seed", "coverage"], as_index=False)["combined_wins"].mean()
    coverage = (
        coverage_df.groupby("coverage", as_index=False)["combined_wins"]
        .agg(combined_win_rate_mean="mean", combined_win_rate_std="std")
        .fillna(0.0)
        .to_dict(orient="records")
    )
    verdict = _seed_verdict(overall_rate, winning_seed_count, len(seeds))
    summary = {
        "command": f"python scripts/run_seed_sweep.py --config {config_path} --seeds {' '.join(map(str, seeds))} --output {output}",
        "seeds": seeds,
        "verdict": verdict,
        "winning_seed_count": winning_seed_count,
        "aggregate": aggregate,
        "split_level": split,
        "coverage_level": coverage,
        "failures": failures,
    }
    (out_dir / "seed_sweep_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    report_name = "cstr_seed_sweep_report.md" if base_config.get("system_id") == "cstr" else "seed_sweep_report.md"
    report_path = Path("reports") / report_name
    report_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_df = pd.DataFrame(
        [{"metric": metric, **values} for metric, values in aggregate.items()]
    )
    report = f"""# Seed Sweep Report

## Command

```bash
{summary["command"]}
```

## Seeds

{", ".join(map(str, seeds))}

## Per-seed verdict

{_table(audit_by_seed, ["seed", "verdict", "overall_combined_win_rate", "best_simple_judge", "notes"])}

## Aggregate result

{_table(aggregate_df, ["metric", "mean", "std", "min", "max"])}

## Combined judge robustness

Combined_linear reached the >= 0.70 per-seed win-rate threshold on {winning_seed_count} of {len(seeds)} seeds.

## Split-level robustness

{_table(pd.DataFrame(split), ["split", "combined_win_rate_mean", "combined_win_rate_std"])}

## Coverage-level robustness

{_table(pd.DataFrame(coverage), ["coverage", "combined_win_rate_mean", "combined_win_rate_std"])}

## Verdict

{verdict}

## Explanation

Average combined win rate across seeds was {overall_rate:.6f}; variance is reported above.

## Known failures

{"- none" if not failures else json.dumps(failures, indent=2)}
"""
    report_path.write_text(report, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a TwoTank seed robustness sweep.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = run_seed_sweep(args.config, args.seeds, args.output)
    print(summary["verdict"])


if __name__ == "__main__":
    main()
