from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scs.experiments.v2 import run_heat_exchanger_sanity


def test_v2_heat_exchanger_sanity_outputs_valid_benchmark(tmp_path: Path) -> None:
    result = run_heat_exchanger_sanity(
        "configs/v2/v2_scientific_strengthening.yaml",
        tmp_path / "heat_sanity",
    )
    assert result["verdict"] == "VALID_HEAT_EXCHANGER_BENCHMARK"
    assert result["finite"] is True
    assert result["nonconstant"] is True
    assert result["ood_differs_from_id"] is True
    assert result["calibration_test_overlap"]["overlap_count"] == 0
    for name in [
        "data_summary.json",
        "distribution_checks.csv",
        "model_error_checks.csv",
        "event_label_checks.json",
    ]:
        assert (tmp_path / "heat_sanity" / name).exists()
    distribution = pd.read_csv(tmp_path / "heat_sanity" / "distribution_checks.csv")
    assert distribution["ood_differs_from_id"].any()
    payload = json.loads((tmp_path / "heat_sanity" / "event_label_checks.json").read_text(encoding="utf-8"))
    assert payload["verdict"] == "VALID_HEAT_EXCHANGER_BENCHMARK"
