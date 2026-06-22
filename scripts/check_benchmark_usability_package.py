from __future__ import annotations

import argparse

from scs.experiments.benchmark_usability import check_benchmark_usability_package


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the benchmark usability package.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = check_benchmark_usability_package(args.config)
    print(result["verdict"])


if __name__ == "__main__":
    main()
