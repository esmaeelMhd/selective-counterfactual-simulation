from __future__ import annotations

import argparse

from scs.reports.failure_analysis import verify_failed_gate


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify failed v0 gate before failure analysis.")
    parser.add_argument("--results", required=True)
    parser.add_argument("--gate", required=True)
    args = parser.parse_args()
    result = verify_failed_gate(args.results, args.gate)
    print(result["verdict"])


if __name__ == "__main__":
    main()
