from __future__ import annotations

import numpy as np
import pytest

from scs.systems.base import TrajectoryBatch


def _linear_batch() -> TrajectoryBatch:
    rng = np.random.default_rng(0)
    n, horizon, state_dim = 12, 6, 2
    actions = rng.normal(size=(n, horizon, 1))
    disturbances = np.zeros((n, horizon, 1))
    states = np.zeros((n, horizon + 1, state_dim))
    states[:, 0, :] = rng.normal(size=(n, state_dim))
    for time in range(horizon):
        states[:, time + 1, :] = 0.5 * states[:, time, :] + actions[:, time, :]
    return TrajectoryBatch(
        states=states,
        actions=actions,
        disturbances=disturbances,
        scenario_type=["linear"] * n,
        split="model_train",
        system_id="two_tank",
    )


def test_lstm_world_learns_a_linear_step() -> None:
    pytest.importorskip("torch")
    from scs.models.lstm_world import LSTMWorldModel

    train = _linear_batch()
    model = LSTMWorldModel(random_state=0, hidden_dim=16, num_layers=1, epochs=40, learning_rate=1e-2)
    model.fit(train)
    assert model.train_losses[-1] < model.train_losses[0]
    predicted = model.predict_rollout(train.states[0, 0], train.actions[0], train.disturbances[0], history=np.zeros((2, 5)))
    assert predicted.shape == train.states[0].shape
    assert np.isfinite(predicted).all()
    assert np.mean((predicted - train.states[0]) ** 2) < np.mean((train.states[0, :1] - train.states[0]) ** 2)


def test_ensemble_lstm_matches_member_shape() -> None:
    pytest.importorskip("torch")
    from scs.models.ensemble_lstm import EnsembleLSTMModel

    train = _linear_batch()
    model = EnsembleLSTMModel(random_state=0, n_members=2, hidden_dim=8, num_layers=1, epochs=2)
    model.fit(train)
    samples = model.predict_rollout_samples(train.states[0, 0], train.actions[0], train.disturbances[0], n_samples=3)
    assert samples.shape == (3, train.states.shape[1], train.states.shape[2])
    assert np.isfinite(samples).all()


def test_recurrent_world_one_step_loss_drops() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("timesim")
    from scs.models.recurrent_world import RecurrentWorldModel

    train = _linear_batch()
    model = RecurrentWorldModel(
        random_state=0,
        hidden_dim=8,
        latent_dim=4,
        encoder_dim=8,
        epochs=3,
        learning_rate=1e-2,
    )
    model.fit(train)
    assert model.train_losses[-1] < model.train_losses[0]
    predicted = model.predict_rollout(train.states[0, 0], train.actions[0], train.disturbances[0])
    assert predicted.shape == train.states[0].shape
    assert np.isfinite(predicted).all()
