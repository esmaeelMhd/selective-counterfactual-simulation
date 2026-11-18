"""Adapter for ``timesim.models.latent_ssm.LatentSSMWorldModel``.

Training uses its posterior filter and a one-step prior prediction of the
next state. PyTorch and timesim are optional. The confirmatory BSM1 test
split is not used here.
"""

from __future__ import annotations

import numpy as np

from scs.models.base import validate_rollout_inputs
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
    ) -> None:
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
        self.train_losses: list[float] = []

    def fit(self, train: TrajectoryBatch) -> None:
        torch, factory = _require_latent()
        torch.manual_seed(self.random_state)
        states = np.asarray(train.states[:, :-1, :], dtype=np.float32)
        targets = np.asarray(train.states[:, 1:, :], dtype=np.float32)
        actions = np.asarray(train.actions, dtype=np.float32)
        disturbances = np.asarray(train.disturbances, dtype=np.float32)
        self.state_dim = int(states.shape[-1])
        self.action_dim = int(actions.shape[-1])
        self.disturbance_dim = int(disturbances.shape[-1])
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
        self.train_losses = []
        model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            filtered = model.observe(controls, exogenous, observed)
            predicted = _prior_next(model, filtered, controls, exogenous)
            reconstruction = torch.mean((predicted - nxt) ** 2)
            divergence = torch.mean(filtered["kl_terms"])
            loss = reconstruction + self.kl_weight * divergence
            loss.backward()
            optimizer.step()
            self.train_losses.append(float(loss.detach()))
        model.eval()
        self.model = model

    def predict_rollout(
        self,
        initial_state: np.ndarray,
        actions: np.ndarray,
        disturbances: np.ndarray,
        history: np.ndarray | None = None,
    ) -> np.ndarray:
        del history
        torch, _factory = _require_latent()
        if self.model is None or self.action_dim is None or self.disturbance_dim is None:
            raise RuntimeError("model must be fit before rollout")
        initial_state, actions, disturbances = validate_rollout_inputs(initial_state, actions, disturbances)
        self.model.eval()
        horizon = len(actions)
        warmup = np.concatenate(
            [
                np.zeros((1, 1, self.action_dim), dtype=np.float32),
                np.zeros((1, 1, self.disturbance_dim), dtype=np.float32),
                initial_state.astype(np.float32)[None, None, :],
            ],
            axis=-1,
        )
        states = np.empty((horizon + 1, initial_state.shape[0]), dtype=float)
        states[0] = initial_state
        with torch.no_grad():
            latent = self.model.init_state(torch.tensor(warmup))
            for time in range(horizon):
                control = torch.tensor(actions[time].astype(np.float32)[None, :])
                exogenous = torch.tensor(disturbances[time].astype(np.float32)[None, :])
                latent, predicted = self.model.step(latent, control, exogenous)
                states[time + 1] = predicted.numpy()[0]
        return states

    def predict_rollout_samples(
        self,
        initial_state: np.ndarray,
        actions: np.ndarray,
        disturbances: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        mean = self.predict_rollout(initial_state, actions, disturbances)
        return np.repeat(mean[None, :, :], n_samples, axis=0)


def _prior_next(model, filtered: dict, controls, exogenous):
    import torch

    steps = controls.shape[1]
    predicted = []
    for time in range(steps):
        latent = {"h": filtered["deter"][:, time], "z": filtered["stoch"][:, time]}
        _state, prediction = model.step(latent, controls[:, time], exogenous[:, time])
        predicted.append(prediction)
    return torch.stack(predicted, dim=1)
