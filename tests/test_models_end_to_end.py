from __future__ import annotations

import numpy as np

from scs.data.generate import generate_dataset
from scs.experiments.registry import make_model
from scs.metrics.trajectory import rmse


def test_required_models_train_and_predict() -> None:
    dataset = generate_dataset("two_tank", 90, 16, 8, horizon=25, dt=0.1, seed=9)
    train = dataset["train"]
    test = dataset["id_test"]
    model_ids = ["hold_last", "linear_narx", "mlp_state_space"]
    errors = {}
    for idx, model_id in enumerate(model_ids):
        model = make_model(model_id, seed=idx)
        model.fit(train)
        pred = model.predict_rollout(test.states[0, 0], test.actions[0], test.disturbances[0])
        assert pred.shape == test.states[0].shape
        assert np.isfinite(pred).all()
        assert not np.allclose(pred, 0.0)
        samples = model.predict_rollout_samples(test.states[0, 0], test.actions[0], test.disturbances[0], n_samples=3)
        assert samples.shape == (3, test.horizon + 1, train.states.shape[-1])
        assert np.isfinite(samples).all()

        split_errors = []
        for i in range(test.n_trajectories):
            pred_i = model.predict_rollout(test.states[i, 0], test.actions[i], test.disturbances[i])
            split_errors.append(rmse(pred_i, test.states[i]))
        errors[model_id] = float(np.mean(split_errors))

    assert errors["linear_narx"] < errors["hold_last"]

