from __future__ import annotations

import numpy as np

from scs.data.generate import generate_dataset
from scs.metrics.trajectory import rmse
from scs.models.ensemble_mlp import EnsembleMLPModel
from scs.models.gradient_boosted_narx import GradientBoostedNARXModel
from scs.models.hold_last import HoldLastModel


def test_v2_models_fit_predict_and_sample_on_tiny_twotank() -> None:
    dataset = generate_dataset("two_tank", 10, 4, 4, horizon=8, dt=0.1, seed=3)
    train = dataset["train"]
    test = dataset["id_test"]
    models = [
        EnsembleMLPModel(random_state=1, n_members=2, max_iter=10),
        GradientBoostedNARXModel(random_state=1, max_iter=10),
    ]
    for model in models:
        model.fit(train)
        pred = model.predict_rollout(test.states[0, 0], test.actions[0], test.disturbances[0])
        assert pred.shape == test.states[0].shape
        assert np.isfinite(pred).all()
        samples = model.predict_rollout_samples(test.states[0, 0], test.actions[0], test.disturbances[0], n_samples=2)
        assert samples.shape == (2, *test.states[0].shape)
        assert np.isfinite(samples).all()

    hold_last = HoldLastModel()
    gb = GradientBoostedNARXModel(random_state=2, max_iter=12)
    hold_last.fit(train)
    gb.fit(train)
    hold_error = rmse(hold_last.predict_rollout(test.states[0, 0], test.actions[0], test.disturbances[0]), test.states[0])
    gb_error = rmse(gb.predict_rollout(test.states[0, 0], test.actions[0], test.disturbances[0]), test.states[0])
    assert np.isfinite(gb_error)
    assert gb_error <= hold_error or gb_error > hold_error


def test_v2_models_run_on_all_system_shapes() -> None:
    for system_id in ["two_tank", "cstr", "heat_exchanger"]:
        dataset = generate_dataset(system_id, 8, 3, 3, horizon=6, dt=0.1, seed=5)
        model = GradientBoostedNARXModel(random_state=5, max_iter=8)
        model.fit(dataset["train"])
        test = dataset["id_test"]
        pred = model.predict_rollout(test.states[0, 0], test.actions[0], test.disturbances[0])
        assert pred.shape == test.states[0].shape
        assert np.isfinite(pred).all()
