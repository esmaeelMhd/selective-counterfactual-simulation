from __future__ import annotations

import argparse

from scs.experiments.current_status import check_claim_language


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan current docs/reports for unsupported claim language.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--paths", nargs="+", required=True)
    args = parser.parse_args()
    result = check_claim_language(args.manifest, args.paths)
    print(result["verdict"])


if __name__ == "__main__":
    main()
