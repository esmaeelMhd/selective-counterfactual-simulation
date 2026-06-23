from __future__ import annotations

import argparse

from scs.experiments.v2_comparator import make_comparator_decision_gate


def main() -> None:
    parser = argparse.ArgumentParser(description="Make the v2 comparator fairness decision gate.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--statistics", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = make_comparator_decision_gate(args.config, args.selection, args.evaluation, args.statistics, args.output)
    print(result["decision"])


if __name__ == "__main__":
    main()
