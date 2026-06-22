from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from examples.custom_model_example import run_example
from scs.experiments.benchmark_usability import compare_models
from scs.models.user_model import UserSimulatorModel


def test_custom_model_example_writes_public_outputs(tmp_path: Path) -> None:
    summary = run_example(tmp_path / "custom_model")

    assert summary["model_id"] == "damped_linear_user"
    assert (tmp_path / "custom_model" / "custom_model_smoke.json").exists()
    assert (tmp_path / "custom_model" / "custom_model_report.md").exists()
    assert summary["is_evidence_for_current_claim"] is False


def test_custom_model_template_documents_shapes() -> None:
    text = Path("examples/my_model_template.py").read_text(encoding="utf-8")

    assert "class MySimulatorModel" in text
    assert "model_id = \"my_simulator\"" in text
    assert "(horizon + 1, state_dim)" in text
    assert "NotImplementedError" in text


def test_custom_model_shape_validation_and_comparison_path(tmp_path: Path) -> None:
    manifest = Path("results/current_status/evidence_manifest/current_evidence_manifest.json")
    before = hashlib.sha256(manifest.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="shape"):
        UserSimulatorModel.validate_rollout_output(np.zeros((2, 2)), np.zeros(3), np.zeros((3, 1)))

    result = compare_models(
        "configs/experiments/calibrated_two_tank.yaml",
        ["linear_narx"],
        tmp_path / "comparison",
        custom_model="examples/custom_model_example.py:DampedLinearUserModel",
    )

    assert result["verdict"] == "MODEL_COMPARISON_BUILT"
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == before
