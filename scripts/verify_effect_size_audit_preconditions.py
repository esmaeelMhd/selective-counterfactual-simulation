from __future__ import annotations

import argparse

from scs.experiments.effect_audit import verify_effect_size_audit_preconditions


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify effect-size audit preconditions.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = verify_effect_size_audit_preconditions(args.config, args.output)
    print(result["verdict"])


if __name__ == "__main__":
    main()
