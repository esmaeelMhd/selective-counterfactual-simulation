from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from scs.experiments.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete TwoTank smoke benchmark.")
    parser.parse_args()
    summary = run_experiment(
        "configs/experiments/smoke_two_tank.yaml",
        command="python scripts/run_smoke.py",
    )
    print("smoke run complete")
    for name, path in summary["artifacts"].items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
