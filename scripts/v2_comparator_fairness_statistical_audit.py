from __future__ import annotations

import argparse

from scs.experiments.v2_comparator import run_comparator_statistical_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v2 comparator fairness statistical audit.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_comparator_statistical_audit(args.config, args.evaluation, args.output)
    print(result["verdict"])


if __name__ == "__main__":
    main()
