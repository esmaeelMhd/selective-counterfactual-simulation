from __future__ import annotations

import argparse

from scs.experiments.current_status import sync_signal_semantics_status


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync signal semantics status from current evidence manifest.")
    parser.add_argument("--status-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = sync_signal_semantics_status(args.status_manifest, args.output)
    print(result["verdict"])


if __name__ == "__main__":
    main()
