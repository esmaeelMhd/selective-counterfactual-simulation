from __future__ import annotations

import argparse

from scs.experiments.v2_public_hardening import verify_public_benchmark_hardening_preconditions


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify v2 public benchmark hardening preconditions.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = verify_public_benchmark_hardening_preconditions(args.config, args.output)
    print(result["verdict"])


if __name__ == "__main__":
    main()
