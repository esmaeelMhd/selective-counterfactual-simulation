from __future__ import annotations

import argparse

from scs.experiments.benchmark_usability import verify_benchmark_usability_preconditions


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify benchmark usability expansion preconditions.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = verify_benchmark_usability_preconditions(args.config, args.output)
    print(result["verdict"])


if __name__ == "__main__":
    main()
