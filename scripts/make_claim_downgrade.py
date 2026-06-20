from __future__ import annotations

import argparse

from scs.reports.claim_downgrade import make_claim_downgrade


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the v0 claim downgrade report from audit artifacts.")
    parser.parse_args()
    report = make_claim_downgrade()
    print(report["action"])


if __name__ == "__main__":
    main()

