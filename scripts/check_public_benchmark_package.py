from __future__ import annotations

import argparse

from scs.experiments.public_benchmark import check_public_benchmark_package


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the public benchmark package end to end.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = check_public_benchmark_package(args.config)
    print(result["verdict"])


if __name__ == "__main__":
    main()
