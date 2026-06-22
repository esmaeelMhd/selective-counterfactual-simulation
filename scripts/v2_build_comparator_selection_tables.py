from __future__ import annotations

import argparse

from scs.experiments.v2_comparator import build_comparator_selection_tables


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v2 comparator fairness calibration-selection tables.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = build_comparator_selection_tables(args.config, args.output)
    print(result["verdict"])


if __name__ == "__main__":
    main()
