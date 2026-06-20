from __future__ import annotations

import argparse

from scs.experiments.calibrated import run_calibrated_judge


def main() -> None:
    parser = argparse.ArgumentParser(description="Run calibrated refusal judge benchmark.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = run_calibrated_judge(args.config, args.output)
    print(summary["verdict"])


if __name__ == "__main__":
    main()
