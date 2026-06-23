from __future__ import annotations

import argparse

from scs.experiments.v2 import validate_event_targets


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate v2 event-risk target definitions.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--event-config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = validate_event_targets(args.config, args.event_config, args.output)
    print(result["verdict"])


if __name__ == "__main__":
    main()
