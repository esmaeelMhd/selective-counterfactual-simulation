from __future__ import annotations

import argparse

from scs.experiments.v2 import verify_v2_preconditions


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify v2 preconditions before protocol lock.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = verify_v2_preconditions(args.config, args.output)
    print(result["verdict"])


if __name__ == "__main__":
    main()
