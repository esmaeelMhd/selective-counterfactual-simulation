from __future__ import annotations

import numpy as np

from scs.models.base import SimulatorModel


class ModelEnsemble:
    model_id = "model_ensemble"

    def __init__(self, models: list[SimulatorModel]) -> None:
        if not models:
            raise ValueError("models must not be empty")
        self.models = models

    def predict_rollout_samples(
        self,
        initial_state: np.ndarray,
        actions: np.ndarray,
        disturbances: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        predictions = [
            model.predict_rollout(initial_state, actions, disturbances)
            for model in self.models
        ]
        stacked = np.asarray(predictions, dtype=float)
        if n_samples <= len(predictions):
            return stacked[:n_samples]
        repeats = int(np.ceil(n_samples / len(predictions)))
        return np.tile(stacked, (repeats, 1, 1))[:n_samples]

