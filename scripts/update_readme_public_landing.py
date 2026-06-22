from __future__ import annotations

import argparse

from scs.experiments.public_benchmark import update_readme_public_landing, write_quickstart_doc


def main() -> None:
    parser = argparse.ArgumentParser(description="Update or check the public README landing section.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--readme", required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = update_readme_public_landing(args.config, args.readme, check=args.check)
    if not args.check:
        write_quickstart_doc()
    print(result["verdict"])


if __name__ == "__main__":
    main()
