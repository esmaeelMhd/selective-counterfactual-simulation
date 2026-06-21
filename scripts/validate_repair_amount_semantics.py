from __future__ import annotations

import argparse

from scs.experiments.repair_signal_semantics import validate_repair_amount_semantics


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate repair_amount semantics with controlled cases.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = validate_repair_amount_semantics(args.config, args.output)
    print(result["verdict"])


if __name__ == "__main__":
    main()
