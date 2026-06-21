from __future__ import annotations

import argparse

from scs.experiments.cstr_weakness import make_cstr_weakness_diagnosis


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize CSTR weakness diagnosis.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = make_cstr_weakness_diagnosis(args.input_dir, args.output)
    print(result["final_diagnosis"])


if __name__ == "__main__":
    main()
