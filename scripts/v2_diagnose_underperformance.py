from __future__ import annotations

import argparse

from scs.experiments.v2 import diagnose_v2_calibrated_underperformance


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose v2 calibrated judge underperformance.")
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = diagnose_v2_calibrated_underperformance(args.results, args.output)
    print(result["verdict"])


if __name__ == "__main__":
    main()
