"""Deep ensemble of LSTM world models. Members are the uncertainty samples."""

from __future__ import annotations

import numpy as np

from scs.models.base import validate_rollout_batch, validate_rollout_inputs
from scs.models.lstm_world import LSTMWorldModel
from scs.systems.base import TrajectoryBatch


class EnsembleLSTMModel:
    model_id = "ensemble_lstm"

    def __init__(
        self,
        random_state: int = 0,
        n_members: int = 5,
        hidden_dim: int = 64,
        num_layers: int = 2,
        epochs: int = 30,
        learning_rate: float = 1e-3,
    ) -> None:
        self.random_state = int(random_state)
        self.n_members = int(n_members)
        if self.n_members < 2:
            raise ValueError("ensemble_lstm needs at least 2 members")
        self.members = [
            LSTMWorldModel(
                random_state=random_state + 101 * index,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                epochs=epochs,
                learning_rate=learning_rate,
            )
            for index in range(self.n_members)
        ]

    def fit(self, train: TrajectoryBatch) -> None:
        for member in self.members:
            member.fit(train)

    def predict_rollout(
        self,
        initial_state: np.ndarray,
        actions: np.ndarray,
        disturbances: np.ndarray,
        history: np.ndarray | None = None,
    ) -> np.ndarray:
        predictions = [
            member.predict_rollout(initial_state, actions, disturbances, history=history) for member in self.members
        ]
        return np.mean(np.stack(predictions, axis=0), axis=0)

    def predict_rollouts(
        self,
        initial_states: np.ndarray,
        actions: np.ndarray,
        disturbances: np.ndarray,
    ) -> np.ndarray:
        initial_states, actions, disturbances = validate_rollout_batch(initial_states, actions, disturbances)
        stacked = [member.predict_rollouts(initial_states, actions, disturbances) for member in self.members]
        return np.mean(np.stack(stacked, axis=0), axis=0)

    def predict_rollout_samples(
        self,
        initial_state: np.ndarray,
        actions: np.ndarray,
        disturbances: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        initial_state, actions, disturbances = validate_rollout_inputs(initial_state, actions, disturbances)
        samples = []
        for index in range(n_samples):
            member = self.members[index % len(self.members)]
            samples.append(member.predict_rollout(initial_state, actions, disturbances))
        return np.stack(samples, axis=0)
