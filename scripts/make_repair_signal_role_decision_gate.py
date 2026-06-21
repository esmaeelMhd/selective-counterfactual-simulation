from __future__ import annotations

import argparse

from scs.experiments.repair_signal_semantics import make_repair_signal_role_decision_gate


def main() -> None:
    parser = argparse.ArgumentParser(description="Make repair signal role decision gate.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--repair-validation", required=True)
    parser.add_argument("--repair-vs-invariant", required=True)
    parser.add_argument("--signal-ablation", required=True)
    parser.add_argument("--seed-sweep", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = make_repair_signal_role_decision_gate(
        args.config,
        args.repair_validation,
        args.repair_vs_invariant,
        args.signal_ablation,
        args.seed_sweep,
        args.output,
    )
    print(result["decision"])


if __name__ == "__main__":
    main()
