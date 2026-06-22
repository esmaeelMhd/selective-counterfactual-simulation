from __future__ import annotations

import argparse

from scs.experiments.technical_note_package import build_release_package_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build release package manifest and release note.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = build_release_package_manifest(args.config, args.output)
    print(result["release_type"])


if __name__ == "__main__":
    main()
