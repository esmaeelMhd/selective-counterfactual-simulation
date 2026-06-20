from __future__ import annotations

import argparse
import json
from pathlib import Path

from scs.data.generate import generate_and_save_dataset
from scs.data.schemas import save_dataset
from scs.experiments.runner import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate selective simulation datasets.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(str(config.get("output_dir", f"results/{config['experiment_id']}")))
    if "systems" in config:
        output_dir.mkdir(parents=True, exist_ok=True)
        data_summary = {}
        for system_entry in config["systems"]:
            if isinstance(system_entry, str):
                system_id = system_entry
                overrides = {}
            else:
                system_id = str(system_entry["system_id"])
                overrides = {key: value for key, value in system_entry.items() if key != "system_id"}
            system_config = {
                key: value
                for key, value in config.items()
                if key not in {"systems", "system_id", "output_dir", "legacy_data_dir"}
            }
            system_config.update(overrides)
            system_config["system_id"] = system_id
            system_dataset = generate_and_save_dataset(system_config, output_dir / system_id)
            data_summary[system_id] = {
                split: {
                    "n_trajectories": batch.n_trajectories,
                    "horizon": batch.horizon,
                    "scenario_type": ",".join(sorted(set(batch.scenario_type))),
                }
                for split, batch in system_dataset.items()
            }
            print(f"generated {system_id} data under {output_dir / system_id / 'data'}")
        with (output_dir / "data_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(data_summary, handle, indent=2, sort_keys=True)
        return

    dataset = generate_and_save_dataset(config, output_dir)
    legacy_data_dir = config.get("legacy_data_dir")
    if legacy_data_dir:
        save_dataset(dataset, legacy_data_dir)
    print(f"generated data under {output_dir / 'data'}")
    if legacy_data_dir:
        print(f"generated compatibility data under {legacy_data_dir}")


if __name__ == "__main__":
    main()
