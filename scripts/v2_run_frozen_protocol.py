from __future__ import annotations

import argparse

from scs.experiments.v2 import run_v2_frozen_protocol


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen v2 scientific-strengthening protocol.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--event-config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_v2_frozen_protocol(args.config, args.event_config, args.output)
    print(result["verdict"])


if __name__ == "__main__":
    main()
