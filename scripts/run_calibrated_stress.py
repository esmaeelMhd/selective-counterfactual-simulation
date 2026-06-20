from __future__ import annotations

import argparse

from scs.experiments.calibrated import run_calibrated_stress


def main() -> None:
    parser = argparse.ArgumentParser(description="Run calibrated threshold/coverage stress test.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--thresholds", nargs="+", type=float, required=True)
    parser.add_argument("--coverages", nargs="+", type=float, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = run_calibrated_stress(args.config, args.thresholds, args.coverages, args.seeds, args.output)
    print(summary["verdict"])


if __name__ == "__main__":
    main()
