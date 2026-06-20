from __future__ import annotations

import argparse

from scs.reports.failure_analysis import analyze_score_ablation


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze combined-score ablations and calibration.")
    parser.add_argument("--failure-table", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = analyze_score_ablation(args.failure_table, args.output_dir)
    print(summary["verdict"])


if __name__ == "__main__":
    main()
