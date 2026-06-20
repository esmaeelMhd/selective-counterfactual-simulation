from __future__ import annotations

import argparse

from scs.reports.failure_analysis import analyze_threshold_sensitivity


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze threshold sensitivity.")
    parser.add_argument("--failure-table", required=True)
    parser.add_argument("--thresholds", nargs="+", type=float, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = analyze_threshold_sensitivity(args.failure_table, args.thresholds, args.output_dir)
    print(summary["verdict"])


if __name__ == "__main__":
    main()
