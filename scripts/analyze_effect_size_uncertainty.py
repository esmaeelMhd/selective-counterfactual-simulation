from __future__ import annotations

import argparse

from scs.experiments.effect_audit import analyze_effect_size_uncertainty


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze calibrated refusal effect size and uncertainty.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = analyze_effect_size_uncertainty(args.config, args.output)
    print(result["verdict"])


if __name__ == "__main__":
    main()
