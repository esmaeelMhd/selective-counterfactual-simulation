from __future__ import annotations

import argparse

from scs.experiments.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete TwoTank smoke benchmark.")
    parser.add_argument("--config", default="configs/experiments/smoke_two_tank.yaml")
    args = parser.parse_args()
    summary = run_experiment(
        args.config,
        command=f"python scripts/run_smoke.py --config {args.config}" if args.config != "configs/experiments/smoke_two_tank.yaml" else "python scripts/run_smoke.py",
    )
    print("smoke run complete")
    for name, path in summary["artifacts"].items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
