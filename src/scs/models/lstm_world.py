"""Adapter around ``timesim.models.lstm.LSTMWorldModel``.

The network is that class. This module standardizes a trajectory batch and
calls it. PyTorch and timesim are optional.

The model is recurrent in training and in rollout. Training runs over whole
trajectories in chunks, teacher-forced on the true previous state, and carries
the hidden and cell state from one chunk to the next, starting from zero. A
rollout starts from the same zero state and carries it step by step, feeding
back its own prediction. ``history`` is accepted and ignored.
"""

from __future__ import annotations

import numpy as np

from scs.models.base import validate_history, validate_rollout_batch, validate_rollout_inputs
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
        batch_trajectories: int = 16,
        chunk_length: int = 48,
    ) -> None:
        if batch_trajectories < 1 or chunk_length < 1:
            raise ValueError("batch_trajectories and chunk_length must be positive")
        self.batch_trajectories = int(batch_trajectories)
        self.chunk_length = int(chunk_length)
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
        n_rows, length, _features = x_tensor.shape
        generator = torch.Generator().manual_seed(self.random_state)
        self.train_losses = []
        net.train()
        for _ in range(self.epochs):
            order = torch.randperm(n_rows, generator=generator)
            total = 0.0
            weight = 0
            for start in range(0, n_rows, self.batch_trajectories):
                choice = order[start : start + self.batch_trajectories]
                hidden = torch.zeros(net.num_layers, int(choice.shape[0]), net.hidden_dim)
                cell = torch.zeros(net.num_layers, int(choice.shape[0]), net.hidden_dim)
                for first in range(0, length, self.chunk_length):
                    stop = min(first + self.chunk_length, length)
                    optimizer.zero_grad()
                    output, (hidden, cell) = net.lstm(x_tensor[choice, first:stop], (hidden, cell))
                    loss = loss_fn(net.fc(output), y_tensor[choice, first:stop])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    optimizer.step()
                    hidden, cell = hidden.detach(), cell.detach()
                    count = int(choice.shape[0]) * (stop - first)
                    total += float(loss.detach()) * count
                    weight += count
            self.train_losses.append(total / weight)
        net.eval()
        with torch.no_grad():
            output, _state = net.lstm(x_tensor)
            predicted = net.fc(output).numpy()
        restored = predicted * self.y_std + self.y_mean
        residual = targets.reshape(-1, targets.shape[-1]) - restored.reshape(-1, restored.shape[-1])
        self.residual_std = np.maximum(np.std(residual, axis=0), 1e-5)
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
        history: np.ndarray | None = None,
        noise: np.ndarray | None = None,
        project=None,
    ) -> np.ndarray:
        """Free-running rollout that carries the hidden and cell state.

        ``noise`` has shape (batch, horizon, state_dim), in state units. It is
        added to each predicted state before that state is fed back.
        ``project(step, states)`` may overwrite channels of that state, and the
        overwritten state is what the next step sees.
        """
        torch, _timesim_lstm = _require_lstm()
        if self.net is None or self.x_mean is None or self.y_mean is None or self.x_std is None or self.y_std is None:
            raise RuntimeError("model must be fit before rollout")
        initial_states, actions, disturbances = validate_rollout_batch(initial_states, actions, disturbances)
        validate_history(history, initial_states.shape[0])
        self.net.eval()
        batch, horizon, action_dim = actions.shape
        disturbance_dim = disturbances.shape[-1]
        state_dim = initial_states.shape[1]
        if noise is not None and noise.shape != (batch, horizon, state_dim):
            raise ValueError("noise must have shape (batch, horizon, state_dim)")
        states = np.empty((batch, horizon + 1, state_dim), dtype=float)
        states[:, 0, :] = initial_states
        controls = torch.tensor(self._scale_columns(actions, 0, action_dim).astype(np.float32))
        exogenous = torch.tensor(
            self._scale_columns(disturbances, action_dim, action_dim + disturbance_dim).astype(np.float32)
        )
        hidden = torch.zeros(self.net.num_layers, batch, self.net.hidden_dim)
        cell = torch.zeros(self.net.num_layers, batch, self.net.hidden_dim)
        previous = torch.tensor(self._scale_state(initial_states).astype(np.float32))
        with torch.no_grad():
            for time in range(horizon):
                (hidden, cell), step = self.net.step((hidden, cell), controls[:, time], exogenous[:, time], previous)
                raw = step.numpy().astype(float) * self.y_std + self.y_mean
                if noise is not None:
                    raw = raw + noise[:, time]
                if project is not None:
                    raw = np.asarray(project(time, raw), dtype=float)
                states[:, time + 1, :] = raw
                previous = torch.tensor(((raw - self.y_mean) / self.y_std).astype(np.float32))
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
