from __future__ import annotations

import argparse

from scs.reports.failure_analysis import make_failure_diagnosis


def main() -> None:
    parser = argparse.ArgumentParser(description="Create final failure diagnosis report.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = make_failure_diagnosis(args.input_dir, args.output)
    print(report["diagnosis"])


if __name__ == "__main__":
    main()
