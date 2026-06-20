from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

from scs.data.generate import generate_dataset
from scs.data.splits import action_range, max_inflow


def test_severity_changes_action_and_disturbance_distributions() -> None:
    low = generate_dataset("two_tank", 20, 6, 6, horizon=12, dt=0.1, seed=3, severity="low", include_pump_degradation=True)
    high = generate_dataset("two_tank", 20, 6, 6, horizon=12, dt=0.1, seed=3, severity="high", include_pump_degradation=True)
    assert action_range(high["ood_action_magnitude"]) > action_range(low["ood_action_magnitude"])
    assert max_inflow(high["ood_inflow_spike"]) > max_inflow(low["ood_inflow_spike"])


def test_severity_sweep_tiny_run(tmp_path) -> None:
    config = yaml.safe_load(Path("configs/experiments/smoke_two_tank.yaml").read_text())
    config.update({"n_train": 24, "n_id_test": 6, "n_ood_test": 6, "horizon": 12, "uncertainty_samples": 3})
    config_path = tmp_path / "tiny.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = tmp_path / "severity"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_severity_sweep.py",
            "--config",
            str(config_path),
            "--severities",
            "low",
            "high",
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert (output / "severity_summary.csv").exists()
    assert (output / "risk_coverage_by_severity.csv").exists()
