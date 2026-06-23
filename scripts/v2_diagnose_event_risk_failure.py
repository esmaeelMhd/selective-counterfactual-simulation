from __future__ import annotations

import argparse

from scs.experiments.v2_event_risk import run_v2_event_risk_fix


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose and test a scoped v2 event-risk repair candidate.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_v2_event_risk_fix(args.config, args.output)
    print(result["verdict"])


if __name__ == "__main__":
    main()
