from __future__ import annotations

import numpy as np


class MySimulatorModel:
    """Minimal custom simulator template.

    Expected shapes:
    - train_batch.states: (n_trajectories, horizon + 1, state_dim)
    - train_batch.actions: (n_trajectories, horizon, action_dim)
    - train_batch.disturbances: (n_trajectories, horizon, disturbance_dim)
    - initial_state: (state_dim,)
    - actions: (horizon, action_dim)
    - disturbances: (horizon, disturbance_dim)
    - predict_rollout return: (horizon + 1, state_dim)

    Custom model results are local-only and are not added to the current
    benchmark evidence claim.
    """

    model_id = "my_simulator"

    def fit(self, train_batch) -> None:
        """Fit on the provided training trajectory batch."""
        raise NotImplementedError("fit(train_batch) must train your simulator")

    def predict_rollout(self, initial_state, actions, disturbances):
        """Return a rollout with shape (horizon + 1, state_dim)."""
        initial_state = np.asarray(initial_state, dtype=float)
        actions = np.asarray(actions, dtype=float)
        _ = np.asarray(disturbances, dtype=float)
        expected_shape = (len(actions) + 1, len(initial_state))
        raise NotImplementedError(f"return an array with shape {expected_shape}")
