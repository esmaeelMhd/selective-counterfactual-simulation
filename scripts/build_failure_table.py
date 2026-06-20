from __future__ import annotations

import argparse

from scs.reports.failure_analysis import build_failure_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical failure-analysis table.")
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    schema = build_failure_table(args.results, args.output)
    print(schema["row_count"])


if __name__ == "__main__":
    main()
