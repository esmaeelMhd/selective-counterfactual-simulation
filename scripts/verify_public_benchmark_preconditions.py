from __future__ import annotations

import argparse

from scs.experiments.public_benchmark import verify_public_benchmark_preconditions


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify public benchmark packaging preconditions.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = verify_public_benchmark_preconditions(args.config, args.output)
    print(result["verdict"])


if __name__ == "__main__":
    main()
