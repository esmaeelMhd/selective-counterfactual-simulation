from __future__ import annotations

import argparse

from scs.experiments.v2_public_hardening import check_public_benchmark_package


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the v2 public benchmark package.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    result = check_public_benchmark_package(args.config, args.manifest)
    print(result["verdict"])


if __name__ == "__main__":
    main()
