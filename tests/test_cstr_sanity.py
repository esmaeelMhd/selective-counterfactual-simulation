from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from conftest import write_tiny_calibrated_cstr_config
from scs.experiments.calibrated import generate_calibrated_data, load_calibrated_config
from scs.experiments.cstr_replication import _label_checks, run_cstr_sanity


def test_cstr_sanity_checks_finite_nonconstant_shifted_data(tmp_path: Path) -> None:
    config_path = write_tiny_calibrated_cstr_config(tmp_path / "tiny_cstr.yaml")
    result = run_cstr_sanity(config_path, tmp_path / "sanity")
    data_summary = (tmp_path / "sanity" / "cstr_data_summary.json").read_text(encoding="utf-8")
    distribution = pd.read_csv(tmp_path / "sanity" / "cstr_distribution_checks.csv")
    assert "finite" in data_summary
    assert result["verdict"] in {"VALID_CSTR_BENCHMARK", "WEAK_CSTR_BENCHMARK", "INVALID_CSTR_BENCHMARK"}
    assert result["distribution_passed"] is True
    assert set(distribution["scenario_type"].str.split(",").str[0]) >= {"id", "cooling_step_change"}

    config = load_calibrated_config(config_path)
    dataset = generate_calibrated_data(config, tmp_path / "data_check")
    for batch in dataset.values():
        assert np.isfinite(batch.states).all()
        assert np.linalg.norm(np.diff(batch.states, axis=1), axis=-1).mean() > 0.0


def test_cstr_label_degeneracy_is_detected() -> None:
    table = pd.DataFrame({"rmse": [0.01, 0.02, 0.03]})
    result = _label_checks(
        {"bad_threshold": {"value": 0.15}},
        table.assign(scenario_type="id"),
        table.assign(scenario_type="id"),
    )
    assert result["labels_non_degenerate"] is False
