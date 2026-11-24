"""Deep ensemble of LSTM world models. Members are the uncertainty samples."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from scs.models.base import validate_rollout_batch, validate_rollout_inputs
from scs.models.lstm_world import LSTMWorldModel
from scs.parallel import configure_worker_threads, resolve_workers
from scs.systems.base import TrajectoryBatch

_WORKER_TRAIN: TrajectoryBatch | None = None


def _init_ensemble_lstm_worker(train: TrajectoryBatch) -> None:
    global _WORKER_TRAIN
    configure_worker_threads(configure_torch=True)
    _WORKER_TRAIN = train


def _fit_lstm_member(member: LSTMWorldModel) -> LSTMWorldModel:
    if _WORKER_TRAIN is None:
        raise RuntimeError("ensemble LSTM worker was not initialized with training data")
    member.fit(_WORKER_TRAIN)
    return member


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
        workers: int = 1,
        **member_options,
    ) -> None:
        """``member_options`` go to every ``LSTMWorldModel``, such as the early-stopping settings."""
        self.random_state = int(random_state)
        self.n_members = int(n_members)
        self.workers = resolve_workers(workers)
        if self.n_members < 2:
            raise ValueError("ensemble_lstm needs at least 2 members")
        self.members = [
            LSTMWorldModel(
                random_state=random_state + 101 * (index + 1),
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                epochs=epochs,
                learning_rate=learning_rate,
                **member_options,
            )
            for index in range(self.n_members)
        ]

    def fit(self, train: TrajectoryBatch, workers: int | None = None) -> None:
        effective_workers = self.workers if workers is None else resolve_workers(workers)
        if effective_workers <= 1 or len(self.members) <= 1:
            for member in self.members:
                member.fit(train)
            return
        n_procs = min(effective_workers, len(self.members))
        with ProcessPoolExecutor(
            max_workers=n_procs,
            initializer=_init_ensemble_lstm_worker,
            initargs=(train,),
        ) as pool:
            self.members = list(pool.map(_fit_lstm_member, self.members))

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
        history: np.ndarray | None = None,
        project=None,
    ) -> np.ndarray:
        initial_states, actions, disturbances = validate_rollout_batch(initial_states, actions, disturbances)
        stacked = [
            member.predict_rollouts(initial_states, actions, disturbances, history=history, project=project)
            for member in self.members
        ]
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
