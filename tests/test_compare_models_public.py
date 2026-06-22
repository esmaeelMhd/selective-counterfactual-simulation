from __future__ import annotations

from pathlib import Path

import pandas as pd

from scs.experiments.benchmark_usability import compare_models


def test_public_builtin_model_comparison_runs(tmp_path: Path) -> None:
    output = tmp_path / "model_comparison"
    result = compare_models("configs/experiments/calibrated_two_tank.yaml", ["hold_last", "linear_narx"], output)

    assert result["verdict"] == "MODEL_COMPARISON_BUILT"
    table = pd.read_csv(output / "model_comparison.csv")
    assert set(table["model_id"]) == {"hold_last", "linear_narx"}
    assert (output / "risk_coverage_by_model.png").stat().st_size > 0


def test_public_custom_model_comparison_marks_custom_model(tmp_path: Path) -> None:
    output = tmp_path / "model_comparison_custom"
    result = compare_models(
        "configs/experiments/calibrated_two_tank.yaml",
        ["linear_narx"],
        output,
        custom_model="examples/custom_model_example.py:DampedLinearUserModel",
    )

    assert result["verdict"] == "MODEL_COMPARISON_BUILT"
    table = pd.read_csv(output / "model_comparison.csv")
    assert table["is_custom"].any()
