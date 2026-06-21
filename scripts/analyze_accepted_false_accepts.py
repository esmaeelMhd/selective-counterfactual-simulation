from __future__ import annotations

import argparse

from scs.experiments.effect_audit import analyze_accepted_false_accepts


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze accepted false accepts for calibrated refusal judges.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = analyze_accepted_false_accepts(args.config, args.output)
    print(result["verdict"])


if __name__ == "__main__":
    main()
