from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import write_tiny_calibrated_config
from examples.custom_model_example import DampedLinearUserModel
from scs.data.generate import generate_dataset
from scs.experiments.benchmark_usability import compare_models


def test_custom_model_example_fits_and_predicts() -> None:
    dataset = generate_dataset("two_tank", 8, 3, 3, horizon=8, dt=0.1, seed=5)
    model = DampedLinearUserModel()
    model.fit(dataset["train"])
    batch = dataset["id_test"]
    pred = model.predict_rollout(batch.states[0, 0], batch.actions[0], batch.disturbances[0])

    assert pred.shape == batch.states[0].shape
    assert np.isfinite(pred).all()


def test_custom_model_invalid_shape_raises() -> None:
    with pytest.raises(ValueError, match="rollout"):
        DampedLinearUserModel.validate_rollout_output(np.zeros((2, 2)), np.zeros(3), np.zeros((3, 1)))


def test_custom_model_runs_through_comparison_path(tmp_path: Path) -> None:
    config = write_tiny_calibrated_config(tmp_path / "tiny.yaml")
    result = compare_models(
        config,
        ["linear_narx"],
        tmp_path / "comparison",
        custom_model="examples/custom_model_example.py:DampedLinearUserModel",
    )

    assert result["verdict"] == "MODEL_COMPARISON_BUILT"
    assert (tmp_path / "comparison" / "model_comparison.csv").exists()
