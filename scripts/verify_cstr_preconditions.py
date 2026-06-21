from __future__ import annotations

import argparse

from scs.experiments.cstr_replication import verify_cstr_preconditions


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify frozen-protocol CSTR replication preconditions.")
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = verify_cstr_preconditions(args.protocol, args.output)
    print(result["verdict"])


if __name__ == "__main__":
    main()
