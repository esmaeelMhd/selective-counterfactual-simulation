"""Adapter for ``timesim.models.latent_ssm.LatentSSMWorldModel``.

Training filters whole trajectories in chunks and predicts the next state from
each filtered latent state. The latent state is carried from one chunk to the
next, starting from zero. A rollout starts from the same zero state, filters
its own previous prediction at every step, and carries the latent state. The
confirmatory BSM1 test split is not used here. ``history`` is accepted and
ignored.

The decoder predicts the change in state over one step, scaled per channel,
and the rollout adds it to the previous state. See ``lstm_world`` for why.
"""

from __future__ import annotations

import numpy as np

from scs.models.base import (
    EarlyStopping,
    subset_batch,
    time_blocked_split,
    validate_history,
    validate_rollout_batch,
    validate_rollout_inputs,
    validation_rmse,
)
from scs.systems.base import TrajectoryBatch


def _require_latent():
    try:
        import torch
        from timesim.models.latent_ssm import LatentSSMWorldModel
    except ImportError as exc:
        raise ImportError(
            "recurrent world model needs torch and timesim"
        ) from exc
    return torch, LatentSSMWorldModel


def _detach(state):
    from timesim.models.rssm import RSSMState

    return RSSMState(h=state.h.detach(), z=state.z.detach())


class RecurrentWorldModel:
    model_id = "recurrent_world"

    def __init__(
        self,
        random_state: int = 0,
        hidden_dim: int = 32,
        latent_dim: int = 8,
        encoder_dim: int = 16,
        epochs: int = 20,
        learning_rate: float = 1e-3,
        kl_weight: float = 0.01,
        batch_trajectories: int = 16,
        chunk_length: int = 48,
        early_stopping: bool = False,
        validation_fraction: float = 0.15,
        validation_gap: int = 0,
        check_every: int = 5,
        patience: int = 3,
        projection_factory=None,
    ) -> None:
        if batch_trajectories < 1 or chunk_length < 1 or check_every < 1:
            raise ValueError("batch_trajectories, chunk_length, and check_every must be positive")
        self.early_stopping = bool(early_stopping)
        self.validation_fraction = float(validation_fraction)
        self.validation_gap = int(validation_gap)
        self.check_every = int(check_every)
        self.patience = int(patience)
        self.projection_factory = projection_factory
        self.selected_epochs: int | None = None
        self.validation_curve: list[tuple[int, float]] = []
        self.batch_trajectories = int(batch_trajectories)
        self.chunk_length = int(chunk_length)
        self.random_state = int(random_state)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.encoder_dim = int(encoder_dim)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.kl_weight = float(kl_weight)
        self.model = None
        self.state_dim: int | None = None
        self.action_dim: int | None = None
        self.disturbance_dim: int | None = None
        self.state_mean: np.ndarray | None = None
        self.state_std: np.ndarray | None = None
        self.delta_mean: np.ndarray | None = None
        self.delta_std: np.ndarray | None = None
        self.action_mean: np.ndarray | None = None
        self.action_std: np.ndarray | None = None
        self.disturbance_mean: np.ndarray | None = None
        self.disturbance_std: np.ndarray | None = None
        self.residual_std: np.ndarray | None = None
        self.train_losses: list[float] = []

    def fit(self, train: TrajectoryBatch) -> None:
        """Train for ``epochs`` epochs, or stop earlier when ``early_stopping`` is set.

        See ``LSTMWorldModel.fit``. Only ``train`` is read.
        """
        torch, factory = _require_latent()
        torch.manual_seed(self.random_state)
        validation = None
        if self.early_stopping:
            fit_rows, validation_rows = time_blocked_split(train.n_trajectories, self.validation_fraction, self.validation_gap)
            validation = subset_batch(train, validation_rows)
            train = subset_batch(train, fit_rows)
        stopper = EarlyStopping(self.patience)
        states = np.asarray(train.states[:, :-1, :], dtype=np.float32)
        targets = np.asarray(train.states[:, 1:, :], dtype=np.float32)
        actions = np.asarray(train.actions, dtype=np.float32)
        disturbances = np.asarray(train.disturbances, dtype=np.float32)
        self.state_dim = int(states.shape[-1])
        self.action_dim = int(actions.shape[-1])
        self.disturbance_dim = int(disturbances.shape[-1])
        self.state_mean = states.mean(axis=(0, 1))
        self.state_std = np.maximum(states.std(axis=(0, 1)), 1e-6)
        self.action_mean = actions.mean(axis=(0, 1))
        self.action_std = np.maximum(actions.std(axis=(0, 1)), 1e-6)
        self.disturbance_mean = disturbances.mean(axis=(0, 1))
        self.disturbance_std = np.maximum(disturbances.std(axis=(0, 1)), 1e-6)
        deltas = targets - states
        self.delta_mean = deltas.mean(axis=(0, 1))
        self.delta_std = np.maximum(deltas.std(axis=(0, 1)), 1e-6)
        states = (states - self.state_mean) / self.state_std
        targets = (deltas - self.delta_mean) / self.delta_std
        actions = (actions - self.action_mean) / self.action_std
        disturbances = (disturbances - self.disturbance_mean) / self.disturbance_std
        model = factory(
            input_dim=self.action_dim + self.disturbance_dim + self.state_dim,
            output_dim=self.state_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            encoder_dim=self.encoder_dim,
            control_dim=self.action_dim,
            exogenous_dim=self.disturbance_dim,
            decoder_layers=1,
            min_std=1e-3,
            max_std=2.0,
            prior_min_std=0.1,
            prior_max_std=1.5,
            posterior_min_std=0.1,
            posterior_max_std=1.5,
            use_symlog=False,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        observed = torch.tensor(states)
        nxt = torch.tensor(targets)
        controls = torch.tensor(actions)
        exogenous = torch.tensor(disturbances)
        n_rows, length, _dim = observed.shape
        generator = torch.Generator().manual_seed(self.random_state)
        self.train_losses = []
        self.model = model
        model.train()
        for epoch in range(self.epochs):
            order = torch.randperm(n_rows, generator=generator)
            total = 0.0
            weight = 0
            for start in range(0, n_rows, self.batch_trajectories):
                choice = order[start : start + self.batch_trajectories]
                carried = None
                for first in range(0, length, self.chunk_length):
                    stop = min(first + self.chunk_length, length)
                    optimizer.zero_grad()
                    filtered = model.observe(
                        controls[choice, first:stop],
                        exogenous[choice, first:stop],
                        observed[choice, first:stop],
                        initial_state=carried,
                    )
                    predicted = _prior_next(model, filtered, controls[choice, first:stop], exogenous[choice, first:stop])
                    reconstruction = torch.mean((predicted - nxt[choice, first:stop]) ** 2)
                    divergence = torch.mean(filtered["kl_terms"])
                    loss = reconstruction + self.kl_weight * divergence
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    carried = _detach(filtered["state"])
                    count = int(choice.shape[0]) * (stop - first)
                    total += float(loss.detach()) * count
                    weight += count
            self.train_losses.append(total / weight)
            if validation is not None and ((epoch + 1) % self.check_every == 0 or epoch + 1 == self.epochs):
                score = validation_rmse(self, validation, self.projection_factory)
                model.train()
                snapshot = lambda: {key: value.clone() for key, value in model.state_dict().items()}
                if stopper.update(epoch + 1, score, snapshot):
                    break
        if validation is not None:
            if stopper.best_state is None:
                raise RuntimeError("validation score never improved; the rollout may have diverged")
            model.load_state_dict(stopper.best_state)
            self.selected_epochs = stopper.best_epoch
            self.validation_curve = stopper.curve
        else:
            self.selected_epochs = self.epochs
        model.eval()
        with torch.no_grad():
            filtered = model.observe(controls, exogenous, observed)
            predicted = _prior_next(model, filtered, controls, exogenous).numpy()
        residual = (predicted - targets) * self.delta_std
        self.residual_std = np.maximum(np.std(residual.reshape(-1, self.state_dim), axis=0), 1e-5)
        self.model = model

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
        """Free-running rollout that carries the latent state.

        ``noise`` has shape (batch, horizon, state_dim), in state units, and is
        added to each predicted state before it is filtered again.
        ``project(step, states)`` may overwrite channels of that state.
        """
        torch, _factory = _require_latent()
        if self.model is None or self.state_mean is None or self.action_mean is None or self.disturbance_mean is None:
            raise RuntimeError("model must be fit before rollout")
        initial_states, actions, disturbances = validate_rollout_batch(initial_states, actions, disturbances)
        validate_history(history, initial_states.shape[0])
        self.model.eval()
        batch, horizon, _action_dim = actions.shape
        if noise is not None and noise.shape != (batch, horizon, initial_states.shape[-1]):
            raise ValueError("noise must have shape (batch, horizon, state_dim)")
        scaled_actions = ((actions - self.action_mean) / self.action_std).astype(np.float32)
        scaled_disturbances = ((disturbances - self.disturbance_mean) / self.disturbance_std).astype(np.float32)
        states = np.empty((batch, horizon + 1, initial_states.shape[-1]), dtype=float)
        states[:, 0, :] = initial_states
        controls = torch.tensor(scaled_actions)
        exogenous = torch.tensor(scaled_disturbances)
        previous = torch.tensor(((initial_states - self.state_mean) / self.state_std).astype(np.float32))
        carried = None
        with torch.no_grad():
            for time in range(horizon):
                filtered = self.model.observe(
                    controls[:, time : time + 1],
                    exogenous[:, time : time + 1],
                    previous[:, None, :],
                    initial_state=carried,
                )
                carried = filtered["state"]
                latent = {"h": filtered["deter"][:, 0], "z": filtered["stoch"][:, 0]}
                _latent, step = self.model.step(latent, controls[:, time], exogenous[:, time])
                raw = states[:, time, :] + step.numpy().astype(float) * self.delta_std + self.delta_mean
                if noise is not None:
                    raw = raw + noise[:, time]
                if project is not None:
                    raw = np.asarray(project(time, raw), dtype=float)
                states[:, time + 1, :] = raw
                previous = torch.tensor(((raw - self.state_mean) / self.state_std).astype(np.float32))
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


def _prior_next(model, filtered: dict, controls, exogenous):
    import torch

    steps = controls.shape[1]
    predicted = []
    for time in range(steps):
        latent = {"h": filtered["deter"][:, time], "z": filtered["stoch"][:, time]}
        _state, prediction = model.step(latent, controls[:, time], exogenous[:, time])
        predicted.append(prediction)
    return torch.stack(predicted, dim=1)
