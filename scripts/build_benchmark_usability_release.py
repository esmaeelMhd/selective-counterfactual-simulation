from __future__ import annotations

import argparse

from scs.experiments.benchmark_usability import build_benchmark_usability_release


def main() -> None:
    parser = argparse.ArgumentParser(description="Build benchmark usability release note and manifest.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = build_benchmark_usability_release(args.config, args.output)
    print(result["release_type"])


if __name__ == "__main__":
    main()
