from __future__ import annotations

import argparse

from scs.experiments.current_status import verify_current_status_preconditions


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify current-status sync preconditions.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = verify_current_status_preconditions(args.config, args.output)
    print(result["verdict"])


if __name__ == "__main__":
    main()
