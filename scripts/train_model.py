from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from _bootstrap import add_src_to_path

add_src_to_path()

from scs.data.generate import generate_and_save_dataset
from scs.experiments.registry import make_model
from scs.experiments.runner import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one simulator model from an experiment config.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config.")
    parser.add_argument("--model", required=True, help="Model id to train.")
    parser.add_argument("--system", help="System id to train when the config contains multiple systems.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.model not in config["models"]:
        raise SystemExit(f"model {args.model!r} is not listed in config models")

    if "systems" in config:
        if not args.system:
            raise SystemExit("--system is required when training from a multi-system config")
        matching = [
            entry for entry in config["systems"]
            if (entry == args.system) or (isinstance(entry, dict) and entry.get("system_id") == args.system)
        ]
        if not matching:
            raise SystemExit(f"system {args.system!r} is not listed in config systems")
        overrides = matching[0] if isinstance(matching[0], dict) else {}
        config = {
            key: value
            for key, value in config.items()
            if key not in {"systems", "system_id", "output_dir", "legacy_data_dir"}
        }
        config.update({key: value for key, value in overrides.items() if key != "system_id"})
        config["system_id"] = args.system
        config["output_dir"] = f"results/{config['experiment_id']}/{args.system}"

    output_dir = Path(str(config.get("output_dir", f"results/{config['experiment_id']}")))
    dataset = generate_and_save_dataset(config, output_dir)
    model = make_model(args.model, seed=int(config["seed"]))
    model.fit(dataset["train"])
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"{args.model}.joblib"
    joblib.dump(model, path)
    print(f"trained {args.model} -> {path}")


if __name__ == "__main__":
    main()
