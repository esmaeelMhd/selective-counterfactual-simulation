from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from scs.models.base import flatten_supervised, validate_rollout_inputs
from scs.systems.base import TrajectoryBatch


class MLPStateSpaceModel:
    model_id = "mlp_state_space"

    def __init__(self, random_state: int = 0) -> None:
        self.random_state = random_state
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.regressor = MLPRegressor(
            hidden_layer_sizes=(48, 24),
            activation="tanh",
            solver="adam",
            learning_rate_init=0.003,
            alpha=1e-4,
            batch_size=256,
            max_iter=300,
            early_stopping=True,
            n_iter_no_change=18,
            random_state=random_state,
        )
        self.residual_std: np.ndarray | None = None

    def fit(self, train: TrajectoryBatch) -> None:
        features, targets = flatten_supervised(train)
        x_scaled = self.x_scaler.fit_transform(features)
        y_scaled = self.y_scaler.fit_transform(targets)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            self.regressor.fit(x_scaled, y_scaled)
        predicted = self.y_scaler.inverse_transform(self.regressor.predict(x_scaled))
        residuals = targets - predicted
        self.residual_std = np.maximum(np.std(residuals, axis=0), 1e-5)

    def _predict_delta(self, state: np.ndarray, action: np.ndarray, disturbance: np.ndarray) -> np.ndarray:
        feature = np.concatenate([state, action, disturbance], axis=0)[None, :]
        x_scaled = self.x_scaler.transform(feature)
        y_scaled = self.regressor.predict(x_scaled)
        return np.asarray(self.y_scaler.inverse_transform(y_scaled)[0], dtype=float)

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
        rng = np.random.default_rng(self.random_state + 1997)
        samples = np.empty((n_samples, len(actions) + 1, initial_state.shape[0]), dtype=float)
        for sample_idx in range(n_samples):
            states = np.empty((len(actions) + 1, initial_state.shape[0]), dtype=float)
            states[0] = initial_state
            for t in range(len(actions)):
                noise = rng.normal(0.0, self.residual_std)
                states[t + 1] = states[t] + self._predict_delta(states[t], actions[t], disturbances[t]) + noise
            samples[sample_idx] = states
        return samples

