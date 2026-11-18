"""Write the BSM1 train and calibration splits. The confirmatory test is not written."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scs.data.schemas import save_dataset
from scs.systems.bsm1 import generate_bsm1_pretest_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the BSM1 dry-weather pretest dataset.")
    parser.add_argument("--output", default="results/v3/bsm1_pretest")
    args = parser.parse_args()
    dataset, manifest = generate_bsm1_pretest_dataset()
    output = Path(args.output)
    save_dataset(dataset, output)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {sorted(dataset)} to {output}")


if __name__ == "__main__":
    main()
