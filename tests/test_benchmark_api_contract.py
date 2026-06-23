from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scs.data.generate import generate_dataset
from scs.models.user_model import (
    load_user_model_from_spec,
    validate_rollout_shape,
    validate_user_model_interface,
)


class MissingFit:
    model_id = "missing_fit"

    def predict_rollout(self, initial_state, actions, disturbances):
        return np.zeros((len(actions) + 1, len(initial_state)))


class MissingPredict:
    model_id = "missing_predict"

    def fit(self, train_batch) -> None:
        return None


class WrongShape:
    model_id = "wrong_shape"

    def fit(self, train_batch) -> None:
        return None

    def predict_rollout(self, initial_state, actions, disturbances):
        return np.zeros((len(actions), len(initial_state)))


class NanModel(WrongShape):
    model_id = "nan_model"

    def predict_rollout(self, initial_state, actions, disturbances):
        out = np.zeros((len(actions) + 1, len(initial_state)))
        out[0, 0] = np.nan
        return out


def test_valid_custom_model_loader_and_validation() -> None:
    model = load_user_model_from_spec("examples/custom_model_example.py:DampedLinearUserModel")
    batch = generate_dataset("two_tank", 4, 2, 2, 6, 0.1, 3)["train"]
    validate_user_model_interface(model, batch)


def test_missing_methods_fail() -> None:
    with pytest.raises(TypeError):
        validate_user_model_interface(MissingFit())
    with pytest.raises(TypeError):
        validate_user_model_interface(MissingPredict())


def test_wrong_shape_and_nan_fail() -> None:
    batch = generate_dataset("two_tank", 4, 2, 2, 6, 0.1, 3)["train"]
    with pytest.raises(ValueError):
        validate_user_model_interface(WrongShape(), batch)
    with pytest.raises(ValueError):
        validate_user_model_interface(NanModel(), batch)
    with pytest.raises(ValueError):
        validate_rollout_shape(np.zeros((2, 2)), batch.states[0, 0], batch.actions[0])


def test_loader_import_errors_are_loud(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad_model.py"
    bad_file.write_text("class Other: pass\n", encoding="utf-8")
    with pytest.raises(ImportError):
        load_user_model_from_spec(f"{bad_file}:Missing")
