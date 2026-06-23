from __future__ import annotations

import argparse

from scs.experiments.public_release import check_public_release_ready


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether the repository is ready for public release.")
    parser.add_argument("--output", default="results/public_release_audit")
    args = parser.parse_args()
    result = check_public_release_ready(args.output)
    print(result["verdict"])
    if result["verdict"] != "PUBLIC_RELEASE_READY":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
