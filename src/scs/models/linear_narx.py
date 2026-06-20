from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from scs.models.base import flatten_supervised, validate_rollout_inputs
from scs.systems.base import TrajectoryBatch


class LinearNARXModel:
    model_id = "linear_narx"

    def __init__(self, alpha: float = 1e-3, random_state: int = 0) -> None:
        self.alpha = alpha
        self.random_state = random_state
        self.regressor = Ridge(alpha=alpha)
        self.residual_std: np.ndarray | None = None
        self.state_dim: int | None = None

    def fit(self, train: TrajectoryBatch) -> None:
        features, targets = flatten_supervised(train)
        self.regressor.fit(features, targets)
        residuals = targets - self.regressor.predict(features)
        self.residual_std = np.maximum(np.std(residuals, axis=0), 1e-5)
        self.state_dim = int(targets.shape[-1])

    def _predict_delta(self, state: np.ndarray, action: np.ndarray, disturbance: np.ndarray) -> np.ndarray:
        feature = np.concatenate([state, action, disturbance], axis=0)[None, :]
        return np.asarray(self.regressor.predict(feature)[0], dtype=float)

    def predict_rollout(
        self,
        initial_state: np.ndarray,
        actions: np.ndarray,
        disturbances: np.ndarray,
    ) -> np.ndarray:
        initial_state, actions, disturbances = validate_rollout_inputs(initial_state, actions, disturbances)
        states = np.empty((len(actions) + 1, initial_state.shape[0]), dtype=float)
        states[0] = initial_state
        for t in range(len(actions)):
            states[t + 1] = states[t] + self._predict_delta(states[t], actions[t], disturbances[t])
        return states

    def predict_rollout_samples(
        self,
        initial_state: np.ndarray,
        actions: np.ndarray,
        disturbances: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        if self.residual_std is None:
            raise RuntimeError("model must be fit before sampling")
        rng = np.random.default_rng(self.random_state + 991)
        samples = np.empty((n_samples, len(actions) + 1, initial_state.shape[0]), dtype=float)
        for sample_idx in range(n_samples):
            states = np.empty((len(actions) + 1, initial_state.shape[0]), dtype=float)
            states[0] = initial_state
            for t in range(len(actions)):
                noise = rng.normal(0.0, self.residual_std)
                states[t + 1] = states[t] + self._predict_delta(states[t], actions[t], disturbances[t]) + noise
            samples[sample_idx] = states
        return samples

