from __future__ import annotations

import argparse
import shlex

from scs.experiments.v2_public_hardening import run_public_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the public selective counterfactual simulation benchmark.")
    parser.add_argument("--model", default=None, help="Custom model spec as file.py:ClassName")
    parser.add_argument("--models", nargs="*", default=None, help="Built-in model ids, e.g. linear_narx mlp_state_space")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    command = "python scripts/run_benchmark.py " + " ".join(shlex.quote(arg) for arg in [
        *(["--model", args.model] if args.model else []),
        *(["--models", *(args.models or [])] if args.models else []),
        "--output",
        args.output,
    ])
    result = run_public_benchmark(args.output, custom_model=args.model, models=args.models, command=command)
    print(result["verdict"])


if __name__ == "__main__":
    main()
