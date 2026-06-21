from __future__ import annotations

import argparse

from scs.experiments.effect_audit import analyze_event_risk


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze event-risk false accepts.")
    parser.add_argument("--effect-config", required=True)
    parser.add_argument("--event-config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = analyze_event_risk(args.effect_config, args.event_config, args.output)
    print(result["verdict"])


if __name__ == "__main__":
    main()
