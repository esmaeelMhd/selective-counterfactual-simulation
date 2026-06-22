from __future__ import annotations

from pathlib import Path

import pandas as pd

from scs.experiments.v2 import run_v2_frozen_protocol
from v2_fixtures import write_tiny_v2_config


def test_tiny_v2_frozen_protocol_runs_end_to_end(tmp_path: Path) -> None:
    config = write_tiny_v2_config(tmp_path / "tiny.yaml")
    output = tmp_path / "frozen"
    result = run_v2_frozen_protocol(config, "configs/v2/v2_event_targets.yaml", output)
    assert result["verdict"] == "V2_FROZEN_PROTOCOL_COMPLETE"
    assert result["valid_systems"] == ["two_tank", "cstr"]
    for name in [
        "v2_scenario_scores.csv",
        "v2_risk_coverage.csv",
        "v2_model_metrics.csv",
        "v2_event_metrics.csv",
        "v2_oracle_gap.csv",
        "v2_run_summary.json",
    ]:
        assert (output / name).exists(), name
        assert (output / name).stat().st_size > 0, name
    risk = pd.read_csv(output / "v2_risk_coverage.csv")
    assert set(risk["system_id"]) == {"two_tank", "cstr"}
    assert set(risk["model_id"]) == {"hold_last", "linear_narx"}
    assert risk["seed"].nunique() == 2
    assert not risk.isna().any().any()
