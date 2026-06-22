from __future__ import annotations

import argparse

from scs.experiments.public_benchmark import build_failure_gallery


def main() -> None:
    parser = argparse.ArgumentParser(description="Build public failure-gallery examples from frozen artifacts.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--figure-dir", required=True)
    args = parser.parse_args()
    result = build_failure_gallery(args.config, args.output, args.figure_dir)
    print(result["verdict"])


if __name__ == "__main__":
    main()
