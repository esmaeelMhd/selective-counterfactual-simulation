from __future__ import annotations

import argparse

from scs.reports.claim_audit import write_claim_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit combined_linear against the strongest simple judge.")
    parser.add_argument("--results", required=True, help="Results directory containing risk_coverage.csv.")
    parser.add_argument("--report", default="reports/v0_claim_audit.md", help="Markdown report path.")
    args = parser.parse_args()
    summary = write_claim_audit(args.results, args.report)
    print(summary["verdict"])


if __name__ == "__main__":
    main()

