from __future__ import annotations

import argparse

from scs.reports.failure_analysis import analyze_benchmark_sanity


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze benchmark sanity for failure analysis.")
    parser.add_argument("--results", required=True)
    parser.add_argument("--failure-table", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = analyze_benchmark_sanity(args.results, args.failure_table, args.output_dir)
    print(summary["verdict"])


if __name__ == "__main__":
    main()
