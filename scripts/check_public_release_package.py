from __future__ import annotations

import argparse

from scs.experiments.public_release import check_public_release_package


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full public release package check.")
    parser.add_argument("--output", default="results/public_release_package_check")
    args = parser.parse_args()
    result = check_public_release_package(args.output)
    print(result["verdict"])
    if result["verdict"] != "PUBLIC_RELEASE_PACKAGE_ACCEPTED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
