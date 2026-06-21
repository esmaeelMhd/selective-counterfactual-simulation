from __future__ import annotations

import argparse

from scs.experiments.current_status import build_current_evidence_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build current evidence manifest.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = build_current_evidence_manifest(args.config, args.output)
    print(result["controlling_claim_label"])


if __name__ == "__main__":
    main()
