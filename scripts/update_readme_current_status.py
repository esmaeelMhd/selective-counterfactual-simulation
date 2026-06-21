from __future__ import annotations

import argparse

from scs.experiments.current_status import update_readme_current_status


def main() -> None:
    parser = argparse.ArgumentParser(description="Update or check README current status block.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--readme", required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = update_readme_current_status(args.manifest, args.readme, check=args.check)
    print(result["verdict"])


if __name__ == "__main__":
    main()
