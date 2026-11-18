"""Adapter around ``timesim.models.lstm.LSTMWorldModel``.

The network is that class. This module only standardizes a trajectory batch
and calls it. PyTorch and timesim are optional.
"""

from __future__ import annotations

import numpy as np

from scs.models.base import validate_rollout_batch, validate_rollout_inputs
from scs.systems.base import TrajectoryBatch


def _require_lstm():
    try:
        import torch
        from timesim.models.lstm import LSTMWorldModel as TimesimLSTM
    except ImportError as exc:
        raise ImportError("LSTM world model needs torch and timesim") from exc
    return torch, TimesimLSTM


class LSTMWorldModel:
    model_id = "lstm_world"

    def __init__(
        self,
        random_state: int = 0,
        hidden_dim: int = 64,
        num_layers: int = 2,
        epochs: int = 30,
        learning_rate: float = 1e-3,
    ) -> None:
        self.random_state = int(random_state)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.net = None
        self.x_mean: np.ndarray | None = None
        self.x_std: np.ndarray | None = None
        self.y_mean: np.ndarray | None = None
        self.y_std: np.ndarray | None = None
        self.residual_std: np.ndarray | None = None
        self.train_losses: list[float] = []

    def fit(self, train: TrajectoryBatch) -> None:
        torch, timesim_lstm = _require_lstm()
        torch.manual_seed(self.random_state)
        features, targets = _sequence_tensors(train)
        self.x_mean = features.mean(axis=(0, 1))
        self.x_std = np.maximum(features.std(axis=(0, 1)), 1e-6)
        state_dim = int(targets.shape[-1])
        self.y_mean = self.x_mean[-state_dim:]
        self.y_std = self.x_std[-state_dim:]
        x_scaled = (features - self.x_mean) / self.x_std
        y_scaled = (targets - self.y_mean) / self.y_std
        action_dim = int(train.actions.shape[-1])
        disturbance_dim = int(train.disturbances.shape[-1])
        net = timesim_lstm(
            input_dim=action_dim + disturbance_dim + state_dim,
            output_dim=state_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            control_dim=action_dim,
            exo_dim=disturbance_dim,
        )
        optimizer = torch.optim.Adam(net.parameters(), lr=self.learning_rate)
        loss_fn = torch.nn.MSELoss()
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32)
        self.train_losses = []
        net.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            hidden, _ = net.lstm(x_tensor)
            predicted = net.fc(hidden)
            loss = loss_fn(predicted, y_tensor)
            loss.backward()
            optimizer.step()
            self.train_losses.append(float(loss.detach()))
        net.eval()
        with torch.no_grad():
            hidden, _ = net.lstm(x_tensor)
            predicted = net.fc(hidden).numpy()
        restored = predicted * self.y_std + self.y_mean
        self.residual_std = np.maximum(np.std(targets - restored, axis=(0, 1)), 1e-5)
        self.net = net

    def predict_rollout(
        self,
        initial_state: np.ndarray,
        actions: np.ndarray,
        disturbances: np.ndarray,
        history: np.ndarray | None = None,
    ) -> np.ndarray:
        del history
        rolled = self.predict_rollouts(initial_state[None, :], actions[None, :, :], disturbances[None, :, :])
        return rolled[0]

    def predict_rollouts(
        self,
        initial_states: np.ndarray,
        actions: np.ndarray,
        disturbances: np.ndarray,
    ) -> np.ndarray:
        torch, _timesim_lstm = _require_lstm()
        if self.net is None or self.x_mean is None or self.y_mean is None or self.x_std is None or self.y_std is None:
            raise RuntimeError("model must be fit before rollout")
        initial_states, actions, disturbances = validate_rollout_batch(initial_states, actions, disturbances)
        self.net.eval()
        batch, horizon, action_dim = actions.shape
        disturbance_dim = disturbances.shape[-1]
        state_dim = initial_states.shape[1]
        states = np.empty((batch, horizon + 1, state_dim), dtype=float)
        states[:, 0, :] = initial_states
        warmup = np.concatenate(
            [
                np.zeros((batch, 1, action_dim), dtype=np.float32),
                np.zeros((batch, 1, disturbance_dim), dtype=np.float32),
                self._scale_state(initial_states).astype(np.float32)[:, None, :],
            ],
            axis=-1,
        )
        controls = self._scale_columns(actions, 0, action_dim)
        exogenous = self._scale_columns(disturbances, action_dim, action_dim + disturbance_dim)
        with torch.no_grad():
            rolled = self.net.rollout(
                warmup_seq={"inputs": torch.tensor(warmup)},
                rollout_inputs={
                    "controls": torch.tensor(controls.astype(np.float32)),
                    "exogenous": torch.tensor(exogenous.astype(np.float32)),
                },
                horizon=horizon,
                feedback="model",
            )
        predicted = rolled["predictions"].numpy() * self.y_std + self.y_mean
        states[:, 1:, :] = predicted
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
        initial_state, actions, disturbances = validate_rollout_inputs(initial_state, actions, disturbances)
        mean = self.predict_rollout(initial_state, actions, disturbances)
        rng = np.random.default_rng(self.random_state + 1997)
        noise = rng.normal(0.0, self.residual_std, size=(n_samples, len(actions), mean.shape[-1]))
        samples = np.repeat(mean[None, :, :], n_samples, axis=0)
        samples[:, 1:, :] = samples[:, 1:, :] + np.cumsum(noise, axis=1)
        return samples


    def _scale_state(self, state: np.ndarray) -> np.ndarray:
        if self.x_mean is None or self.x_std is None:
            raise RuntimeError("model must be fit before rollout")
        offset = self.x_mean.shape[0] - state.shape[-1]
        return (state - self.x_mean[offset:]) / self.x_std[offset:]

    def _scale_columns(self, values: np.ndarray, start: int, stop: int) -> np.ndarray:
        if self.x_mean is None or self.x_std is None:
            raise RuntimeError("model must be fit before rollout")
        return (values - self.x_mean[start:stop]) / self.x_std[start:stop]


def _sequence_tensors(train: TrajectoryBatch) -> tuple[np.ndarray, np.ndarray]:
    states = np.asarray(train.states, dtype=float)
    actions = np.asarray(train.actions, dtype=float)
    disturbances = np.asarray(train.disturbances, dtype=float)
    features = np.concatenate([actions, disturbances, states[:, :-1, :]], axis=-1)
    targets = states[:, 1:, :]
    return features, targets
