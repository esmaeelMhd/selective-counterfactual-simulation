from __future__ import annotations

import argparse

from scs.experiments.current_status import make_current_status_decision_gate


def main() -> None:
    parser = argparse.ArgumentParser(description="Make current-status decision gate.")
    parser.add_argument("--preconditions", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--signal-sync", required=True)
    parser.add_argument("--readme-sync", required=True)
    parser.add_argument("--claim-language", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = make_current_status_decision_gate(
        args.preconditions,
        args.manifest,
        args.signal_sync,
        args.readme_sync,
        args.claim_language,
        args.output,
    )
    print(result["decision"])


if __name__ == "__main__":
    main()
