from __future__ import annotations

import argparse

from scs.experiments.repair_signal_semantics import write_signal_semantics_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Write signal semantics registry report.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    artifacts = write_signal_semantics_report(args.output)
    print(artifacts["report"])


if __name__ == "__main__":
    main()
