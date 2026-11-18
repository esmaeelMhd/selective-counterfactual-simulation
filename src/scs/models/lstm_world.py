"""LSTM world model.

The step input is control, exogenous input, then the previous state. That is
the same split as ``LSTMWorldModel`` in time-series-simulator. This module
does not import that package. PyTorch is optional.
"""

from __future__ import annotations

import numpy as np

from scs.models.base import validate_rollout_batch, validate_rollout_inputs
from scs.systems.base import TrajectoryBatch


def _require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise ImportError("LSTM world model needs the optional extra: pip install torch") from exc
    return torch, nn


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
        torch, nn = _require_torch()
        torch.manual_seed(self.random_state)
        features, targets = _sequence_tensors(train)
        self.x_mean = features.mean(axis=(0, 1))
        self.x_std = np.maximum(features.std(axis=(0, 1)), 1e-6)
        self.y_mean = targets.mean(axis=(0, 1))
        self.y_std = np.maximum(targets.std(axis=(0, 1)), 1e-6)
        x_scaled = (features - self.x_mean) / self.x_std
        y_scaled = (targets - self.y_mean) / self.y_std
        net = nn.LSTM(
            input_size=features.shape[-1],
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
        )
        head = nn.Linear(self.hidden_dim, targets.shape[-1])
        module = nn.ModuleDict({"lstm": net, "head": head})
        optimizer = torch.optim.Adam(module.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32)
        self.train_losses = []
        module.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            hidden, _ = module["lstm"](x_tensor)
            predicted = module["head"](hidden)
            loss = loss_fn(predicted, y_tensor)
            loss.backward()
            optimizer.step()
            self.train_losses.append(float(loss.detach()))
        module.eval()
        with torch.no_grad():
            hidden, _ = module["lstm"](x_tensor)
            predicted = module["head"](hidden).numpy()
        restored = predicted * self.y_std + self.y_mean
        self.residual_std = np.maximum(np.std(targets - restored, axis=(0, 1)), 1e-5)
        self.net = module

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
        torch, _nn = _require_torch()
        if self.net is None or self.x_mean is None or self.y_mean is None or self.x_std is None or self.y_std is None:
            raise RuntimeError("model must be fit before rollout")
        initial_states, actions, disturbances = validate_rollout_batch(initial_states, actions, disturbances)
        self.net.eval()
        batch, horizon, _action_dim = actions.shape
        state_dim = initial_states.shape[1]
        states = np.empty((batch, horizon + 1, state_dim), dtype=float)
        states[:, 0, :] = initial_states
        hidden = None
        current = torch.tensor(initial_states, dtype=torch.float32)
        x_mean = torch.tensor(self.x_mean, dtype=torch.float32)
        x_std = torch.tensor(self.x_std, dtype=torch.float32)
        y_mean = torch.tensor(self.y_mean, dtype=torch.float32)
        y_std = torch.tensor(self.y_std, dtype=torch.float32)
        with torch.no_grad():
            for time in range(horizon):
                action = torch.tensor(actions[:, time, :], dtype=torch.float32)
                disturbance = torch.tensor(disturbances[:, time, :], dtype=torch.float32)
                feature = torch.cat([action, disturbance, current], dim=-1)
                scaled = (feature - x_mean) / x_std
                output, hidden = self.net["lstm"](scaled.unsqueeze(1), hidden)
                predicted = self.net["head"](output.squeeze(1)) * y_std + y_mean
                current = predicted
                states[:, time + 1, :] = predicted.numpy()
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


def _sequence_tensors(train: TrajectoryBatch) -> tuple[np.ndarray, np.ndarray]:
    states = np.asarray(train.states, dtype=float)
    actions = np.asarray(train.actions, dtype=float)
    disturbances = np.asarray(train.disturbances, dtype=float)
    features = np.concatenate([actions, disturbances, states[:, :-1, :]], axis=-1)
    targets = states[:, 1:, :]
    return features, targets
