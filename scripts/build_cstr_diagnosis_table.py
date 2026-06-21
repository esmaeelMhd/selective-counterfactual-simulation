from __future__ import annotations

import argparse

from scs.experiments.cstr_weakness import build_cstr_diagnosis_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical CSTR weakness diagnosis table.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = build_cstr_diagnosis_table(args.config, args.output)
    print(result["verdict"])


if __name__ == "__main__":
    main()
