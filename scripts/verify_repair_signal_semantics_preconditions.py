from __future__ import annotations

import argparse

from scs.experiments.repair_signal_semantics import verify_repair_signal_semantics_preconditions


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify repair-signal semantics audit preconditions.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = verify_repair_signal_semantics_preconditions(args.config, args.output)
    print(result["verdict"])


if __name__ == "__main__":
    main()
