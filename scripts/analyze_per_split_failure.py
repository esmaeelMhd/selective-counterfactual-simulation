from __future__ import annotations

import argparse

from scs.reports.failure_analysis import analyze_per_split_failure


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze failure by split and intervention type.")
    parser.add_argument("--failure-table", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = analyze_per_split_failure(args.failure_table, args.output_dir)
    print(summary["verdict"])


if __name__ == "__main__":
    main()
