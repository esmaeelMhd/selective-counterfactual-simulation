from __future__ import annotations

from pathlib import Path

import pandas as pd

from scs.experiments.v2 import trajectory_bad_event, validate_event_targets
from v2_fixtures import write_tiny_v2_config


def test_v2_event_labels_are_trajectory_based_and_thresholds_explicit(tmp_path: Path) -> None:
    config = write_tiny_v2_config(tmp_path / "tiny.yaml", systems=["cstr", "heat_exchanger"])
    result = validate_event_targets(config, "configs/v2/v2_event_targets.yaml", tmp_path / "events")
    assert result["event_labels_from_trajectories"] is True
    assert result["nondegenerate_system_count"] >= 1
    frame = pd.read_csv(tmp_path / "events" / "event_target_validation.csv")
    assert {"event_positive_rate", "is_degenerate"}.issubset(frame.columns)


def test_v2_event_computation_not_rmse_proxy() -> None:
    states = [[1.0, 340.0], [1.0, 366.0]]
    event_config = {
        "cstr": {
            "thresholds": {
                "temperature_high": 365.0,
                "concentration_low": 0.72,
                "concentration_high": 1.65,
            }
        }
    }
    assert trajectory_bad_event("cstr", states, event_config) is True
