from __future__ import annotations

import argparse

from scs.experiments.benchmark_usability import run_current_status_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the current-status quickstart demo.")
    parser.add_argument("--config", default="configs/status/benchmark_usability_v1_1.yaml")
    parser.add_argument("--output", default="results/demo")
    args = parser.parse_args()
    result = run_current_status_demo(args.config, args.output)
    print(result["verdict"])


if __name__ == "__main__":
    main()
