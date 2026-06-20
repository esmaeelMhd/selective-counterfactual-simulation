from __future__ import annotations

import argparse

from scs.experiments.calibrated import verify_calibrated_preconditions


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify calibrated judge preconditions.")
    parser.add_argument("--config", default="configs/experiments/calibrated_two_tank.yaml")
    args = parser.parse_args()
    result = verify_calibrated_preconditions(args.config)
    print(result["verdict"])


if __name__ == "__main__":
    main()
