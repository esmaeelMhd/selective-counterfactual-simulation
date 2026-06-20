from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from scs.experiments.runner import load_results
from scs.reports.summary import write_smoke_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate the smoke markdown report from result files.")
    parser.add_argument("--results", required=True, help="Result directory containing summary and CSV files.")
    args = parser.parse_args()
    summary, risk_coverage, model_metrics = load_results(args.results)
    report_path = summary.get("artifacts", {}).get("smoke_report", "reports/smoke_report.md")
    write_smoke_report(
        summary,
        risk_coverage,
        model_metrics,
        report_path,
        command=f"python scripts/make_report.py --results {args.results}",
    )
    print(report_path)


if __name__ == "__main__":
    main()

