from __future__ import annotations

import argparse

from scs.experiments.calibrated import make_calibrated_decision_gate


def main() -> None:
    parser = argparse.ArgumentParser(description="Create calibrated judge decision gate.")
    parser.add_argument("--single-run", required=True)
    parser.add_argument("--seed-sweep", required=True)
    parser.add_argument("--stress", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = make_calibrated_decision_gate(args.single_run, args.seed_sweep, args.stress, args.output)
    print(result["decision"])


if __name__ == "__main__":
    main()
