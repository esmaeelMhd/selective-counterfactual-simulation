from __future__ import annotations

import argparse

from scs.experiments.public_release import check_fresh_clone_repro


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a fresh-copy install and reproduction check.")
    parser.add_argument("--repo-path", required=True)
    parser.add_argument("--output", default="results/fresh_clone_check")
    args = parser.parse_args()
    result = check_fresh_clone_repro(args.repo_path, args.output)
    print(result["verdict"])
    if result["verdict"] != "FRESH_CLONE_REPRO_PASSED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
