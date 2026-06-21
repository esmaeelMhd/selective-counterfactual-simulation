from __future__ import annotations

import argparse

from scs.experiments.repair_signal_semantics import run_signal_set_ablation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repair signal-set ablation.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_signal_set_ablation(args.config, args.output)
    print(result["verdict"])


if __name__ == "__main__":
    main()
