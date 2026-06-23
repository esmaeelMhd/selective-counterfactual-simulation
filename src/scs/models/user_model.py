from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Protocol

import numpy as np

from scs.models.base import validate_rollout_inputs
from scs.systems.base import TrajectoryBatch


class UserSimulatorModelProtocol(Protocol):
    """Protocol for user-supplied simulator models.

    Implementations intentionally match the existing ``SimulatorModel`` shape
    contract so custom models can be compared locally without changing the
    benchmark internals.
    """

    model_id: str

    def fit(self, train_batch: TrajectoryBatch) -> None:
        """Fit on a training trajectory batch."""

    def predict_rollout(
        self,
        initial_state: np.ndarray,
        actions: np.ndarray,
        disturbances: np.ndarray,
    ) -> np.ndarray:
        """Return predicted states with shape ``(horizon + 1, state_dim)``."""


class UserSimulatorModel:
    """Base class with validation helpers for bring-your-own models."""

    model_id = "user_model"

    def fit(self, train_batch: TrajectoryBatch) -> None:
        raise NotImplementedError("custom models must implement fit(train_batch)")

    def predict_rollout(
        self,
        initial_state: np.ndarray,
        actions: np.ndarray,
        disturbances: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError("custom models must implement predict_rollout")

    def predict_rollout_samples(
        self,
        initial_state: np.ndarray,
        actions: np.ndarray,
        disturbances: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        rollout = self.predict_rollout(initial_state, actions, disturbances)
        return np.repeat(rollout[None, :, :], int(n_samples), axis=0)

    @staticmethod
    def validate_inputs(
        initial_state: np.ndarray,
        actions: np.ndarray,
        disturbances: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return validate_rollout_inputs(initial_state, actions, disturbances)

    @staticmethod
    def validate_rollout_output(
        rollout: np.ndarray,
        initial_state: np.ndarray,
        actions: np.ndarray,
    ) -> np.ndarray:
        rollout = np.asarray(rollout, dtype=float)
        expected = (len(actions) + 1, len(initial_state))
        if rollout.shape != expected:
            raise ValueError(f"custom model rollout must have shape {expected}, got {rollout.shape}")
        if not np.isfinite(rollout).all():
            raise ValueError("custom model rollout contains non-finite values")
        return rollout


def validate_rollout_shape(
    rollout: np.ndarray,
    initial_state: np.ndarray,
    actions: np.ndarray,
) -> np.ndarray:
    """Validate a user model rollout and return it as a finite float array."""
    rollout = np.asarray(rollout, dtype=float)
    initial_state = np.asarray(initial_state, dtype=float)
    actions = np.asarray(actions, dtype=float)
    expected = (len(actions) + 1, len(initial_state))
    if rollout.shape != expected:
        raise ValueError(f"predict_rollout must return shape {expected}, got {rollout.shape}")
    if not np.isfinite(rollout).all():
        raise ValueError("predict_rollout returned NaN or infinite values")
    return rollout


def validate_user_model_interface(model: object, sample_batch: TrajectoryBatch | None = None) -> None:
    """Fail loudly if a user-supplied model does not satisfy the public contract."""
    if not hasattr(model, "fit") or not callable(getattr(model, "fit")):
        raise TypeError("user model must define fit(train_batch)")
    if not hasattr(model, "predict_rollout") or not callable(getattr(model, "predict_rollout")):
        raise TypeError("user model must define predict_rollout(initial_state, actions, disturbances)")
    if not isinstance(getattr(model, "model_id", None), str) or not getattr(model, "model_id"):
        raise TypeError("user model must define a non-empty string model_id")
    if sample_batch is not None:
        model.fit(sample_batch)
        prediction = model.predict_rollout(
            sample_batch.states[0, 0],
            sample_batch.actions[0],
            sample_batch.disturbances[0],
        )
        validate_rollout_shape(prediction, sample_batch.states[0, 0], sample_batch.actions[0])


def load_user_model_from_spec(spec: str) -> object:
    """Load ``/path/to/file.py:ClassName`` and instantiate the class."""
    if ":" not in spec:
        raise ValueError("custom model spec must be in file.py:ClassName format")
    file_part, class_name = spec.rsplit(":", 1)
    path = Path(file_part)
    if not path.exists():
        raise FileNotFoundError(f"custom model file not found: {path}")
    module_name = f"scs_user_model_{abs(hash(path.resolve()))}"
    module_spec = importlib.util.spec_from_file_location(module_name, path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"cannot load custom model module from {path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    try:
        cls = getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(f"custom model class not found: {class_name}") from exc
    model = cls()
    validate_user_model_interface(model)
    return model
