from __future__ import annotations

import argparse

from scs.experiments.repair_signal_semantics import compare_repair_vs_invariant


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare repair_amount and invariant_residual.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = compare_repair_vs_invariant(args.config, args.output)
    print(result["verdict"])


if __name__ == "__main__":
    main()
