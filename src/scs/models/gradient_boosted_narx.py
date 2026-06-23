from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from scs.models.base import flatten_supervised, validate_rollout_inputs
from scs.systems.base import TrajectoryBatch


class GradientBoostedNARXModel:
    """Gradient-boosted NARX delta model with one regressor per state dimension."""

    model_id = "gradient_boosted_narx"

    def __init__(self, random_state: int = 0, max_iter: int = 80) -> None:
        self.random_state = random_state
        self.max_iter = int(max_iter)
        self.regressors: list[HistGradientBoostingRegressor] = []
        self.residual_std: np.ndarray | None = None
        self.state_dim: int | None = None

    def fit(self, train: TrajectoryBatch) -> None:
        features, targets = flatten_supervised(train)
        self.state_dim = int(targets.shape[-1])
        self.regressors = []
        predictions = []
        for dim in range(self.state_dim):
            regressor = HistGradientBoostingRegressor(
                max_iter=self.max_iter,
                learning_rate=0.06,
                l2_regularization=1e-4,
                random_state=self.random_state + dim,
            )
            regressor.fit(features, targets[:, dim])
            self.regressors.append(regressor)
            predictions.append(regressor.predict(features))
        predicted = np.vstack(predictions).T
        self.residual_std = np.maximum(np.std(targets - predicted, axis=0), 1e-5)

    def _predict_delta(self, state: np.ndarray, action: np.ndarray, disturbance: np.ndarray) -> np.ndarray:
        if not self.regressors:
            raise RuntimeError("model must be fit before prediction")
        feature = np.concatenate([state, action, disturbance], axis=0)[None, :]
        return np.asarray([regressor.predict(feature)[0] for regressor in self.regressors], dtype=float)

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
        rng = np.random.default_rng(self.random_state + 7717)
        samples = np.empty((n_samples, len(actions) + 1, initial_state.shape[0]), dtype=float)
        for sample_idx in range(n_samples):
            states = np.empty((len(actions) + 1, initial_state.shape[0]), dtype=float)
            states[0] = initial_state
            for t in range(len(actions)):
                noise = rng.normal(0.0, self.residual_std)
                states[t + 1] = states[t] + self._predict_delta(states[t], actions[t], disturbances[t]) + noise
            samples[sample_idx] = states
        return samples
