from __future__ import annotations

import argparse

from scs.experiments.calibrated import run_calibrated_seed_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description="Run calibrated refusal judge seed sweep.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = run_calibrated_seed_sweep(args.config, args.seeds, args.output)
    print(summary["verdict"])


if __name__ == "__main__":
    main()
