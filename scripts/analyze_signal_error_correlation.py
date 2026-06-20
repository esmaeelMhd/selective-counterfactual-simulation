from __future__ import annotations

import argparse

from scs.reports.failure_analysis import analyze_signal_error_correlation


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze validator signal/error correlation.")
    parser.add_argument("--failure-table", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = analyze_signal_error_correlation(args.failure_table, args.output_dir)
    print(summary["verdict"])


if __name__ == "__main__":
    main()
