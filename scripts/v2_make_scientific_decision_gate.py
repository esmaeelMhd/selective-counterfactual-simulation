from __future__ import annotations

import argparse

from scs.experiments.v2 import make_v2_scientific_decision_gate


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the v2 scientific decision gate.")
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--heat-sanity", required=True)
    parser.add_argument("--frozen-run", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = make_v2_scientific_decision_gate(
        protocol=args.protocol,
        heat_sanity=args.heat_sanity,
        frozen_run=args.frozen_run,
        stats=args.stats,
        output=args.output,
    )
    print(result["decision"])


if __name__ == "__main__":
    main()
