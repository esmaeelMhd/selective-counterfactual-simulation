from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml


def test_seed_sweep_tiny_two_seed_run(tmp_path) -> None:
    config = yaml.safe_load(Path("configs/experiments/smoke_two_tank.yaml").read_text())
    config.update({"n_train": 24, "n_id_test": 6, "n_ood_test": 6, "horizon": 12, "uncertainty_samples": 3})
    config_path = tmp_path / "tiny.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = tmp_path / "seed_sweep"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_seed_sweep.py",
            "--config",
            str(config_path),
            "--seeds",
            "0",
            "1",
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    for seed in [0, 1]:
        assert (output / f"seed_{seed}" / "risk_coverage.csv").exists()
        assert (output / f"seed_{seed}" / "claim_audit.json").exists()
    audit = pd.read_csv(output / "claim_audit_by_seed.csv")
    assert set(audit["seed"]) == {0, 1}
    assert "status" in audit.columns
    assert (output / "seed_sweep_summary.json").exists()
