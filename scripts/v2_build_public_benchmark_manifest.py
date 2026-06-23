from __future__ import annotations

import argparse

from scs.experiments.v2_public_hardening import build_public_benchmark_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the v2 public benchmark package manifest.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = build_public_benchmark_manifest(args.config, args.output)
    print(result["package_id"])


if __name__ == "__main__":
    main()
