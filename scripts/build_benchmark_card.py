from __future__ import annotations

import argparse

from scs.experiments.benchmark_usability import build_benchmark_card


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the benchmark card.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = build_benchmark_card(args.config, args.manifest, args.output)
    print(result["verdict"])


if __name__ == "__main__":
    main()
