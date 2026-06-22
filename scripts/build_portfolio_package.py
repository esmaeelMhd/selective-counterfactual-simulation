from __future__ import annotations

import argparse

from scs.experiments.technical_note_package import build_portfolio_package


def main() -> None:
    parser = argparse.ArgumentParser(description="Build portfolio package docs.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = build_portfolio_package(args.config, args.manifest, args.output_dir)
    print(result["verdict"])


if __name__ == "__main__":
    main()
