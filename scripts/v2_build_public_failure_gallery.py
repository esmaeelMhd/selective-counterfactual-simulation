from __future__ import annotations

import argparse

from scs.experiments.v2_public_hardening import build_public_failure_gallery


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the public event-risk failure gallery.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--figure-dir", required=True)
    args = parser.parse_args()
    result = build_public_failure_gallery(args.config, args.output, args.figure_dir)
    print(result["verdict"])


if __name__ == "__main__":
    main()
