from __future__ import annotations

import argparse

from scs.experiments.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run selective counterfactual evaluation.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config.")
    args = parser.parse_args()
    summary = run_experiment(
        args.config,
        command=f"python scripts/evaluate_selective.py --config {args.config}",
    )
    print(summary["artifacts"]["risk_coverage"])


if __name__ == "__main__":
    main()
