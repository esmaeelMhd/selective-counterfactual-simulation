from __future__ import annotations

import argparse

from scs.experiments.v3 import run_v3_stage


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v3 Stage A and write one results table and one report.")
    parser.add_argument("--config", default="configs/v3/stage_a.yaml")
    parser.add_argument("--output", default="results/v3/stage_a")
    args = parser.parse_args()
    result = run_v3_stage(args.config, args.output)
    print(
        f"eligible_systems={result['eligible_systems']} precision_pass={result['precision_pass']} "
        f"results={result['results']} report={result['report']}"
    )


if __name__ == "__main__":
    main()
