from __future__ import annotations

import argparse

from scs.experiments.technical_note_package import build_limitations_first_technical_note


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the limitations-first technical note.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--tables", required=True)
    parser.add_argument("--figures", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = build_limitations_first_technical_note(args.config, args.tables, args.figures, args.output)
    print(result["title"])


if __name__ == "__main__":
    main()
