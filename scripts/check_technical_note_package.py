from __future__ import annotations

import argparse

from scs.experiments.technical_note_package import check_technical_note_package


def main() -> None:
    parser = argparse.ArgumentParser(description="Check limitations-first technical note package.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    result = check_technical_note_package(args.config, args.manifest)
    print(result["verdict"])


if __name__ == "__main__":
    main()
