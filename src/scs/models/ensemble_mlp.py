from __future__ import annotations

import numpy as np

from scs.models.base import validate_rollout_inputs
from scs.models.mlp_state_space import MLPStateSpaceModel
from scs.systems.base import TrajectoryBatch


class EnsembleMLPModel:
    """Small ensemble of MLP state-space models for v2 uncertainty checks."""

    model_id = "ensemble_mlp"

    def __init__(self, random_state: int = 0, n_members: int = 2, max_iter: int = 120) -> None:
        self.random_state = random_state
        self.n_members = int(n_members)
        self.max_iter = int(max_iter)
        self.members = [
            MLPStateSpaceModel(
                random_state=random_state + 101 * idx,
                hidden_layer_sizes=(32,),
                max_iter=max_iter,
            )
            for idx in range(self.n_members)
        ]
        self.residual_std: np.ndarray | None = None

    def fit(self, train: TrajectoryBatch) -> None:
        for member in self.members:
            member.fit(train)
        stds = [member.residual_std for member in self.members if member.residual_std is not None]
        self.residual_std = np.mean(np.vstack(stds), axis=0) if stds else np.full(train.state_dim, 1e-5)

    def predict_rollout(
        self,
        initial_state: np.ndarray,
        actions: np.ndarray,
        disturbances: np.ndarray,
    ) -> np.ndarray:
        initial_state, actions, disturbances = validate_rollout_inputs(initial_state, actions, disturbances)
        predictions = [
            member.predict_rollout(initial_state, actions, disturbances)
            for member in self.members
        ]
        return np.mean(np.stack(predictions, axis=0), axis=0)

    def predict_rollout_samples(
        self,
        initial_state: np.ndarray,
        actions: np.ndarray,
        disturbances: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        initial_state, actions, disturbances = validate_rollout_inputs(initial_state, actions, disturbances)
        if not self.members:
            raise RuntimeError("ensemble has no members")
        samples = []
        for sample_idx in range(n_samples):
            member = self.members[sample_idx % len(self.members)]
            samples.append(member.predict_rollout(initial_state, actions, disturbances))
        return np.stack(samples, axis=0)
