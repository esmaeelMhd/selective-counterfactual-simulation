from __future__ import annotations

import argparse

from scs.experiments.public_benchmark import run_smoke_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the public smoke-only benchmark demo.")
    parser.add_argument("--output", default="results/smoke_demo")
    args = parser.parse_args()
    result = run_smoke_demo(args.output)
    print(result["verdict"])


if __name__ == "__main__":
    main()
