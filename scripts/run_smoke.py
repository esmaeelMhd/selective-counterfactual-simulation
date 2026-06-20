from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from scs.experiments.runner import run_experiment


def _report_path_for(config_path: str) -> str:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if config.get("experiment_id") == "smoke_cstr" or config.get("system_id") == "cstr":
        return "reports/cstr_smoke_report.md"
    return "reports/smoke_report.md"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete TwoTank smoke benchmark.")
    parser.add_argument("--config", default="configs/experiments/smoke_two_tank.yaml")
    args = parser.parse_args()
    summary = run_experiment(
        args.config,
        report_path=_report_path_for(args.config),
        command=f"python scripts/run_smoke.py --config {args.config}" if args.config != "configs/experiments/smoke_two_tank.yaml" else "python scripts/run_smoke.py",
    )
    print("smoke run complete")
    for name, path in summary["artifacts"].items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
