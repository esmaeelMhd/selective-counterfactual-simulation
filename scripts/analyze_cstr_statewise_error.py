from __future__ import annotations

import argparse

from scs.experiments.cstr_weakness import analyze_cstr_statewise_error


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze CSTR state-wise errors.")
    parser.add_argument("--diagnosis-table", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = analyze_cstr_statewise_error(args.diagnosis_table, args.config, args.output)
    print(result["verdict"])


if __name__ == "__main__":
    main()
