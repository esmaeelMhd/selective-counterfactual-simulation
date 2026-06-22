from __future__ import annotations

import argparse

from scs.experiments.benchmark_usability import update_readme_usability_sections


def main() -> None:
    parser = argparse.ArgumentParser(description="Update/check README usability sections.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--readme", required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = update_readme_usability_sections(args.config, args.readme, check=args.check)
    print(result["verdict"])


if __name__ == "__main__":
    main()
