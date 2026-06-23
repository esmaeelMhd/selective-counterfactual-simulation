from __future__ import annotations

import argparse

from scs.experiments.v2 import run_heat_exchanger_sanity


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v2 HeatExchanger benchmark sanity checks.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_heat_exchanger_sanity(args.config, args.output)
    print(result["verdict"])


if __name__ == "__main__":
    main()
