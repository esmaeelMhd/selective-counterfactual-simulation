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


def test_lstm_first_rollout_step_matches_training() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("timesim")
    import torch

    from scs.models.lstm_world import LSTMWorldModel, _sequence_tensors

    train = _linear_batch()
    model = LSTMWorldModel(random_state=0, hidden_dim=16, num_layers=1, epochs=5, learning_rate=1e-2)
    model.fit(train)
    features, _targets = _sequence_tensors(train)
    scaled = (features[:, :1, :] - model.x_mean) / model.x_std
    with torch.no_grad():
        hidden, _cell = model.net.lstm(torch.tensor(scaled, dtype=torch.float32))
        first = train.states[:, 0] + model.net.fc(hidden[:, 0]).numpy() * model.y_std + model.y_mean
    rolled = model.predict_rollouts(train.states[:, 0], train.actions[:, :1], train.disturbances[:, :1])
    assert np.allclose(rolled[:, 1], first, atol=1e-5)


def test_lstm_world_learns_a_linear_step() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("timesim")
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
    pytest.importorskip("timesim")
    from scs.models.ensemble_lstm import EnsembleLSTMModel

    train = _linear_batch()
    model = EnsembleLSTMModel(random_state=0, n_members=2, hidden_dim=8, num_layers=1, epochs=2)
    model.fit(train)
    samples = model.predict_rollout_samples(train.states[0, 0], train.actions[0], train.disturbances[0], n_samples=3)
    assert samples.shape == (3, train.states.shape[1], train.states.shape[2])
    assert np.isfinite(samples).all()


def test_recurrent_first_rollout_step_matches_training() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("timesim")
    import torch

    from scs.models.recurrent_world import RecurrentWorldModel, _prior_next

    train = _linear_batch()
    model = RecurrentWorldModel(random_state=0, hidden_dim=8, latent_dim=4, encoder_dim=8, epochs=2, learning_rate=1e-2)
    model.fit(train)
    state = ((train.states[:, 0] - model.state_mean) / model.state_std).astype(np.float32)
    action = ((train.actions[:, :1] - model.action_mean) / model.action_std).astype(np.float32)
    disturbance = ((train.disturbances[:, :1] - model.disturbance_mean) / model.disturbance_std).astype(np.float32)
    with torch.no_grad():
        filtered = model.model.observe(torch.tensor(action), torch.tensor(disturbance), torch.tensor(state[:, None, :]))
        first = _prior_next(model.model, filtered, torch.tensor(action), torch.tensor(disturbance)).numpy()
    restored = train.states[:, 0] + first[:, 0, :] * model.delta_std + model.delta_mean
    rolled = model.predict_rollouts(train.states[:, 0], train.actions[:, :1], train.disturbances[:, :1])
    assert np.allclose(rolled[:, 1], restored, atol=1e-5)


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


def _fitted_lstm():
    from scs.models.lstm_world import LSTMWorldModel

    train = _linear_batch()
    model = LSTMWorldModel(random_state=0, hidden_dim=16, num_layers=1, epochs=30, learning_rate=1e-2, batch_trajectories=4, chunk_length=3)
    model.fit(train)
    return model, train


def _fitted_recurrent():
    from scs.models.recurrent_world import RecurrentWorldModel

    train = _linear_batch()
    model = RecurrentWorldModel(
        random_state=0, hidden_dim=8, latent_dim=4, encoder_dim=8, epochs=15, learning_rate=1e-2, batch_trajectories=4, chunk_length=3
    )
    model.fit(train)
    return model, train


def _chained_one_step(model, train, steps: int) -> np.ndarray:
    """Roll one step at a time and restart the model each step, which drops any recurrent state."""
    state = train.states[:, 0]
    for time in range(steps):
        state = model.predict_rollouts(state, train.actions[:, time : time + 1], train.disturbances[:, time : time + 1])[:, 1]
    return state


@pytest.mark.parametrize("fitted", [_fitted_lstm, _fitted_recurrent])
def test_rollout_carries_the_recurrent_state(fitted) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("timesim")
    model, train = fitted()
    carried = model.predict_rollouts(train.states[:, 0], train.actions[:, :3], train.disturbances[:, :3])[:, 3]
    restarted = _chained_one_step(model, train, 3)
    assert np.isfinite(carried).all()
    assert not np.allclose(carried, restarted, atol=1e-6)


def test_lstm_rollout_equals_one_pass_of_the_lstm_over_its_own_inputs() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("timesim")
    import torch

    model, train = _fitted_lstm()
    horizon = 4
    rolled = model.predict_rollouts(train.states[:, 0], train.actions[:, :horizon], train.disturbances[:, :horizon])
    inputs = np.concatenate([train.actions[:, :horizon], train.disturbances[:, :horizon], rolled[:, :horizon]], axis=-1)
    scaled = (inputs - model.x_mean) / model.x_std
    with torch.no_grad():
        output, _state = model.net.lstm(torch.tensor(scaled, dtype=torch.float32))
        one_pass = rolled[:, :-1] + model.net.fc(output).numpy() * model.y_std + model.y_mean
    assert np.allclose(rolled[:, 1:], one_pass, atol=1e-4)


@pytest.mark.parametrize("fitted", [_fitted_lstm, _fitted_recurrent])
def test_project_and_noise_reach_the_rollout(fitted) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("timesim")
    model, train = fitted()
    initial, actions, disturbances = train.states[:, 0], train.actions[:, :3], train.disturbances[:, :3]
    plain = model.predict_rollouts(initial, actions, disturbances)

    def pin(step: int, states: np.ndarray) -> np.ndarray:
        updated = states.copy()
        updated[:, 0] = 7.0
        return updated

    pinned = model.predict_rollouts(initial, actions, disturbances, project=pin)
    assert np.allclose(pinned[:, 1:, 0], 7.0)
    assert not np.allclose(pinned[:, 1:, 1], plain[:, 1:, 1])
    zero = model.predict_rollouts(initial, actions, disturbances, noise=np.zeros((initial.shape[0], 3, initial.shape[1])))
    assert np.allclose(zero, plain, atol=1e-6)
    noisy = model.predict_rollouts(initial, actions, disturbances, noise=np.full((initial.shape[0], 3, initial.shape[1]), 0.5))
    assert not np.allclose(noisy, plain)


@pytest.mark.parametrize("fitted", [_fitted_lstm, _fitted_recurrent])
def test_history_is_checked_and_does_not_change_the_rollout(fitted) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("timesim")
    model, train = fitted()
    initial, actions, disturbances = train.states[:, 0], train.actions[:, :3], train.disturbances[:, :3]
    plain = model.predict_rollouts(initial, actions, disturbances)
    with_history = model.predict_rollouts(initial, actions, disturbances, history=np.ones((initial.shape[0], 2, 5)))
    assert np.allclose(plain, with_history)
    with pytest.raises(ValueError, match="history"):
        model.predict_rollouts(initial, actions, disturbances, history=np.ones((initial.shape[0] + 1, 2, 5)))


def test_stage_a_runs_with_the_torch_models(tmp_path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("timesim")
    import yaml

    from scs.experiments.v3 import run_v3_stage

    config = {
        "stage": "A",
        "label": "exploratory",
        "event_config": "configs/v2/v2_event_targets.yaml",
        "systems": ["two_tank"],
        "models": ["hold_last", "linear_narx", "lstm_world", "ensemble_lstm", "recurrent_world"],
        "learned_models": ["lstm_world", "ensemble_lstm", "recurrent_world"],
        "eligibility_baselines": ["hold_last", "linear_narx"],
        "primary_coverages": [0.1],
        "badness_targets": ["bad_rmse"],
        "rmse_threshold": {"two_tank": 0.05},
        "seeds": [0],
        "event_guard_inputs": ["risk_invariant_only", "risk_disagreement_only", "risk_support_only"],
        "bootstrap_samples": 10,
        "bootstrap_seed": 1,
        "interval_width_limit": 0.5,
        "min_test_scenarios": 1,
        "data": {
            "horizon": 4,
            "dt": 0.1,
            "n_model_train": 6,
            "n_calibration_id": 3,
            "n_calibration_ood": 3,
            "n_test_id": 4,
            "n_test_ood": 3,
            "uncertainty_samples": 2,
        },
        "model_runtime": {
            "ensemble_members": 5,
            "lstm_hidden": 4,
            "lstm_layers": 1,
            "lstm_epochs": 1,
            "rssm_hidden": 4,
            "rssm_latent": 2,
            "rssm_epochs": 1,
        },
    }
    path = tmp_path / "torch.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    result = run_v3_stage(path, tmp_path / "out")
    assert result["rows"] > 0
    assert (tmp_path / "out" / "stage_a_report.md").is_file()


def test_time_blocked_split_keeps_a_gap_between_fit_and_validation_windows() -> None:
    from scs.models.base import time_blocked_split

    fit, validation = time_blocked_split(160, 0.15, 18)
    assert (fit.start, fit.stop) == (0, 118)
    assert (validation.start, validation.stop) == (136, 160)
    assert validation.start - fit.stop == 18
    with pytest.raises(ValueError, match="too few"):
        time_blocked_split(20, 0.15, 18)
    with pytest.raises(ValueError):
        time_blocked_split(100, 1.5, 0)


def test_early_stopping_keeps_the_best_snapshot_and_stops_after_patience() -> None:
    from scs.models.base import EarlyStopping

    stopper = EarlyStopping(patience=2)
    scores = [5.0, 3.0, 3.5, 3.4, 9.0]
    stopped_at = None
    for epoch, score in enumerate(scores, start=1):
        if stopper.update(epoch, score, lambda epoch=epoch: f"state{epoch}"):
            stopped_at = epoch
            break
    assert stopped_at == 4
    assert stopper.best_epoch == 2 and stopper.best_state == "state2"
    assert stopper.curve[:2] == [(1, 5.0), (2, 3.0)]


def test_bsm1_overlap_gap_and_projection() -> None:
    from scs.systems.bsm1 import FLOW_INDEX, REACTOR_WIDTH, hydraulic_projection, pretest_overlap_gap

    assert pretest_overlap_gap() == 18
    disturbances = np.zeros((2, 3, 21))
    disturbances[:, :, FLOW_INDEX] = 20000.0
    projected = hydraulic_projection(disturbances)(1, np.zeros((2, 136)))
    assert np.allclose(projected[:, FLOW_INDEX], 20000.0 + 4.0 * 18446.0)
    assert np.allclose(projected[:, FLOW_INDEX + 5 * REACTOR_WIDTH], 20000.0 - 385.0)


@pytest.mark.parametrize("kind", ["lstm", "recurrent"])
def test_early_stopping_restores_the_best_weights_and_reads_only_the_training_batch(kind) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("timesim")
    from scs.models.base import subset_batch, time_blocked_split, validation_rmse
    from scs.models.lstm_world import LSTMWorldModel
    from scs.models.recurrent_world import RecurrentWorldModel

    train = _linear_batch()
    common = dict(
        random_state=0, epochs=12, learning_rate=1e-2, batch_trajectories=4, chunk_length=3,
        early_stopping=True, validation_fraction=0.25, validation_gap=1, check_every=3, patience=2,
    )
    if kind == "lstm":
        model = LSTMWorldModel(hidden_dim=8, num_layers=1, **common)
    else:
        model = RecurrentWorldModel(hidden_dim=8, latent_dim=4, encoder_dim=8, **common)
    model.fit(train)
    assert model.validation_curve and model.selected_epochs in {epoch for epoch, _score in model.validation_curve}
    best = min(score for _epoch, score in model.validation_curve)
    _fit_rows, validation_rows = time_blocked_split(train.n_trajectories, 0.25, 1)
    assert validation_rmse(model, subset_batch(train, validation_rows)) == pytest.approx(best, rel=1e-4)
    assert len(model.train_losses) <= 12
    # Without early stopping the full budget is used and nothing is held out.
    plain = LSTMWorldModel(random_state=0, hidden_dim=8, num_layers=1, epochs=4, batch_trajectories=4, chunk_length=3)
    plain.fit(train)
    assert plain.selected_epochs == 4 and len(plain.train_losses) == 4


def test_ensemble_member_zero_has_distinct_seed_from_single_model_and_epochs_validated() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("timesim")
    from scs.models.ensemble_lstm import EnsembleLSTMModel
    from scs.models.ensemble_mlp import EnsembleMLPModel
    from scs.models.lstm_world import LSTMWorldModel
    from scs.models.recurrent_world import RecurrentWorldModel

    ens_lstm = EnsembleLSTMModel(random_state=0, n_members=3, hidden_dim=4, num_layers=1, epochs=1)
    assert ens_lstm.members[0].random_state != 0
    assert len({member.random_state for member in ens_lstm.members}) == 3

    # EnsembleMLPModel preserves default v2 seed_offset=0, and supports non-zero offset for v3.
    ens_mlp_v2 = EnsembleMLPModel(random_state=0, n_members=2, max_iter=5)
    ens_mlp_v3 = EnsembleMLPModel(random_state=0, n_members=2, max_iter=5, seed_offset=101)
    assert ens_mlp_v2.members[0].random_state == 0
    assert ens_mlp_v3.members[0].random_state != 0

    with pytest.raises(ValueError, match="epochs"):
        LSTMWorldModel(epochs=0)
    with pytest.raises(ValueError, match="epochs"):
        RecurrentWorldModel(epochs=0)
    # patience=0 is harmless when early_stopping is disabled.
    no_es = LSTMWorldModel(random_state=0, hidden_dim=4, num_layers=1, epochs=1, early_stopping=False, patience=0)
    no_es.fit(_linear_batch())
    assert no_es.selected_epochs == 1
