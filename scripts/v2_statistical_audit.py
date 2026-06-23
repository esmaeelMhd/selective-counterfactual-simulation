from __future__ import annotations

import argparse

from scs.experiments.v2 import run_v2_statistical_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit v2 effect sizes, CIs, and practical thresholds.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_v2_statistical_audit(args.config, args.results, args.output)
    print(result["verdict"])


if __name__ == "__main__":
    main()
