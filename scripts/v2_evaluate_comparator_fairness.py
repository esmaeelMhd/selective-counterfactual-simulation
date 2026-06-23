from __future__ import annotations

import argparse

from scs.experiments.v2_comparator import evaluate_comparator_fairness


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate v2 comparator fairness against deployable and diagnostic baselines.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--selections", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = evaluate_comparator_fairness(args.config, args.selections, args.output)
    print(result["verdict"])


if __name__ == "__main__":
    main()
