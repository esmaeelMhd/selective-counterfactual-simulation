from __future__ import annotations

import argparse

from scs.experiments.v2_public_hardening import build_public_event_risk_figure


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the public v2 event-risk figure.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = build_public_event_risk_figure(args.config, args.output)
    print(result["verdict"])


if __name__ == "__main__":
    main()
