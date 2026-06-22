from __future__ import annotations

import argparse

from scs.experiments.technical_note_package import build_technical_note_evidence_tables


def main() -> None:
    parser = argparse.ArgumentParser(description="Build limitations-first technical note evidence tables.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = build_technical_note_evidence_tables(args.config, args.output)
    print(result["allowed_claim"])


if __name__ == "__main__":
    main()
