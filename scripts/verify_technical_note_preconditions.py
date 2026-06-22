from __future__ import annotations

import argparse

from scs.experiments.technical_note_package import verify_technical_note_preconditions


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify technical-note package preconditions.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = verify_technical_note_preconditions(args.config, args.output)
    print(result["verdict"])


if __name__ == "__main__":
    main()
