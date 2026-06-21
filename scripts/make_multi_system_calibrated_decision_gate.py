from __future__ import annotations

import argparse

from scs.experiments.cstr_replication import make_multi_system_calibrated_decision_gate


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the multi-system calibrated CSTR replication decision gate.")
    parser.add_argument("--twotank-single", required=True)
    parser.add_argument("--twotank-seed", required=True)
    parser.add_argument("--twotank-stress", required=True)
    parser.add_argument("--cstr-sanity", required=True)
    parser.add_argument("--cstr-single", required=True)
    parser.add_argument("--cstr-seed", required=True)
    parser.add_argument("--cstr-stress", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = make_multi_system_calibrated_decision_gate(
        twotank_single=args.twotank_single,
        twotank_seed=args.twotank_seed,
        twotank_stress=args.twotank_stress,
        cstr_sanity=args.cstr_sanity,
        cstr_single=args.cstr_single,
        cstr_seed=args.cstr_seed,
        cstr_stress=args.cstr_stress,
        output=args.output,
    )
    print(result["decision"])


if __name__ == "__main__":
    main()
