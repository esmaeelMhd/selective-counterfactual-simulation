from __future__ import annotations

import argparse

from scs.experiments.technical_note_package import build_technical_note_figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Build limitations-first technical note figures.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--tables", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = build_technical_note_figures(args.config, args.tables, args.output)
    print(result["verdict"])


if __name__ == "__main__":
    main()
