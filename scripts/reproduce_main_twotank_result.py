from __future__ import annotations

import argparse

from scs.experiments.public_benchmark import reproduce_main_twotank_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the frozen TwoTank low-coverage result.")
    parser.add_argument("--output", default="results/reproduce_twotank")
    args = parser.parse_args()
    result = reproduce_main_twotank_result(args.output)
    print(result["verdict"])


if __name__ == "__main__":
    main()
