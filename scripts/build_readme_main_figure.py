from __future__ import annotations

import argparse

from scs.experiments.public_benchmark import build_readme_main_figure


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the public README low-coverage result figure.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = build_readme_main_figure(args.manifest, args.output)
    print(result["verdict"])


if __name__ == "__main__":
    main()
