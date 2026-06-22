from __future__ import annotations

import argparse

from scs.experiments.benchmark_usability import compare_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare built-in and custom simulator models locally.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--custom-model", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = compare_models(args.config, args.models, args.output, custom_model=args.custom_model)
    print(result["verdict"])


if __name__ == "__main__":
    main()
