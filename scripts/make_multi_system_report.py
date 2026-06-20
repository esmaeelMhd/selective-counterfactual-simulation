from __future__ import annotations

import argparse

from scs.reports.multi_system import make_multi_system_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a multi-system selective-simulation report.")
    parser.add_argument("--results", nargs="+", required=True, help="Per-system result directories.")
    parser.add_argument("--output", required=True, help="Markdown report path.")
    parser.add_argument("--gate", default="reports/v0_decision_gate.json", help="Decision gate JSON path.")
    args = parser.parse_args()
    report = make_multi_system_report(args.results, args.output, gate_path=args.gate)
    print(report["overall_claim_status"])


if __name__ == "__main__":
    main()
