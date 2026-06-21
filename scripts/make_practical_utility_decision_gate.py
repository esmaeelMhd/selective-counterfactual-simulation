from __future__ import annotations

import argparse

from scs.experiments.effect_audit import make_practical_utility_decision_gate


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the practical utility decision gate.")
    parser.add_argument("--effect-size", required=True)
    parser.add_argument("--forensics", required=True)
    parser.add_argument("--event-risk", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = make_practical_utility_decision_gate(
        effect_size=args.effect_size,
        forensics=args.forensics,
        event_risk=args.event_risk,
        output=args.output,
    )
    print(result["decision"])


if __name__ == "__main__":
    main()
