from __future__ import annotations

import argparse

from scs.experiments.cstr_weakness import verify_cstr_weakness_audit_preconditions


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify CSTR weakness audit preconditions.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = verify_cstr_weakness_audit_preconditions(args.config, args.output)
    print(result["verdict"])


if __name__ == "__main__":
    main()
