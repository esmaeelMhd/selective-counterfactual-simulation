from __future__ import annotations

import argparse

from scs.experiments.cstr_replication import run_cstr_sanity


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CSTR benchmark sanity checks before calibrated evidence.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_cstr_sanity(args.config, args.output)
    print(result["verdict"])


if __name__ == "__main__":
    main()
