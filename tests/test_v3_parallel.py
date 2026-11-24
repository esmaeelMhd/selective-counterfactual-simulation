from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from scs.systems.base import TrajectoryBatch


def _synthetic_linear_batch(n: int = 12, horizon: int = 6, state_dim: int = 2) -> TrajectoryBatch:
    rng = np.random.default_rng(0)
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


def _tiny_v3_ode_config(event_config_path: str) -> dict:
    return {
        "stage": "A",
        "label": "exploratory",
        "event_config": event_config_path,
        "systems": ["two_tank"],
        "models": [
            "hold_last",
            "linear_narx",
            "mlp_state_space",
            "ensemble_mlp",
            "lstm_world",
            "ensemble_lstm",
            "recurrent_world",
        ],
        "learned_models": [
            "mlp_state_space",
            "ensemble_mlp",
            "lstm_world",
            "ensemble_lstm",
            "recurrent_world",
        ],
        "eligibility_split": "judge_calibration_id",
        "eligibility_baselines": ["hold_last", "linear_narx"],
        "primary_coverages": [0.25, 0.50],
        "descriptive_coverages": [0.75],
        "badness_targets": ["bad_rmse", "bad_event"],
        "rmse_threshold": {"two_tank": 0.08},
        "event_guards": {
            "event_guard": ["risk_support_only", "risk_invariant_only", "risk_disagreement_only"],
            "event_guard_no_invariant": ["risk_support_only", "risk_uncertainty_only", "risk_disagreement_only"],
        },
        "seeds": [0, 1],
        "bootstrap_samples": 40,
        "bootstrap_seed": 2025,
        "interval_width_limit": 0.50,
        "min_test_scenarios": 8,
        "degenerate_share": 0.01,
        "data": {
            "horizon": 6,
            "dt": 0.1,
            "n_model_train": 10,
            "n_calibration_id": 6,
            "n_calibration_ood": 6,
            "n_test_id": 6,
            "n_test_ood": 6,
            "uncertainty_samples": 5,
        },
        "model_runtime": {
            "ensemble_members": 5,
            "mlp_max_iter": 10,
            "ensemble_mlp_max_iter": 10,
            "lstm_hidden": 8,
            "lstm_layers": 1,
            "lstm_epochs": 2,
            "rssm_hidden": 8,
            "rssm_latent": 4,
            "rssm_epochs": 2,
        },
    }


def test_resolve_workers_handles_auto_explicit_and_invalid() -> None:
    from scs.parallel import resolve_workers

    expected_auto = max(1, os.cpu_count() or 1)
    assert resolve_workers(1) == 1
    assert resolve_workers(4) == 4
    assert resolve_workers(0) == expected_auto
    assert resolve_workers(None) == expected_auto
    with pytest.raises(ValueError, match="workers"):
        resolve_workers(-1)
    with pytest.raises(ValueError, match="workers"):
        resolve_workers(True)


def test_bsm1_pretest_parallel_matches_sequential_bit_for_bit() -> None:
    pytest.importorskip("bsm2_python")
    from scs.systems.bsm1 import generate_bsm1_pretest_dataset, pretest_specs
    from tests.test_v3_bsm1_interface import _state_after_short_rollout

    state = _state_after_short_rollout()
    counts = dict(n_model_train=4, n_calibration_id=2, n_calibration_control=2, horizon=4, history=2)
    starts = [int(spec["start_minute"]) for spec in pretest_specs(seed=0, **counts)]
    start_states = {start: state for start in starts}

    seq_data, seq_manifest = generate_bsm1_pretest_dataset(
        seed=0, start_states=start_states, workers=1, **counts
    )
    par_data, par_manifest = generate_bsm1_pretest_dataset(
        seed=0, start_states=start_states, workers=2, **counts
    )

    assert set(par_data) == set(seq_data)
    assert par_manifest == seq_manifest
    for split in seq_data:
        assert np.array_equal(par_data[split].states, seq_data[split].states)
        assert np.array_equal(par_data[split].actions, seq_data[split].actions)
        assert np.array_equal(par_data[split].disturbances, seq_data[split].disturbances)
        assert np.array_equal(par_data[split].measured_history, seq_data[split].measured_history)
        assert par_data[split].scenario_type == seq_data[split].scenario_type


def test_bsm1_confirmatory_parallel_matches_sequential_bit_for_bit(monkeypatch) -> None:
    pytest.importorskip("bsm2_python")
    from scs.systems import bsm1
    from tests.test_v3_bsm1_interface import _state_after_short_rollout

    state = _state_after_short_rollout()
    monkeypatch.setattr(bsm1, "_cached_start_states", lambda dry, starts: {int(start): state for start in starts})
    monkeypatch.setattr(bsm1, "_cached_rain_start_states", lambda dry, rain, starts: {int(start): state for start in starts})

    seq = bsm1.generate_bsm1_confirmatory_dataset(
        n_per_split=3, horizon=4, history=2, seed=1, rehearsal=True, workers=1
    )
    par = bsm1.generate_bsm1_confirmatory_dataset(
        n_per_split=3, horizon=4, history=2, seed=1, rehearsal=True, workers=2
    )

    assert set(par) == set(seq)
    for split in seq:
        assert np.array_equal(par[split].states, seq[split].states)
        assert np.array_equal(par[split].actions, seq[split].actions)
        assert np.array_equal(par[split].disturbances, seq[split].disturbances)
        assert np.array_equal(par[split].measured_history, seq[split].measured_history)
        assert par[split].scenario_type == seq[split].scenario_type


def test_bsm1_stage_cache_is_reusable_across_worker_counts(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("bsm2_python")
    from scs.systems import bsm1
    from tests.test_v3_bsm1_interface import _state_after_short_rollout

    state = _state_after_short_rollout()
    monkeypatch.setattr(bsm1, "_cached_start_states", lambda dry, starts: {int(start): state for start in starts})
    cfg = {
        "stage": "B-rehearsal",
        "data": {
            "horizon": 4,
            "history_minutes": 2,
            "n_model_train": 3,
            "n_calibration_id": 2,
            "n_calibration_ood": 2,
            "n_test_per_split": 2,
            "cache_dir": str(tmp_path / "cache"),
        },
    }
    first = bsm1.generate_bsm1_pretest_stage_dataset(cfg, seed=0, workers=2)

    def _boom(*args, **kwargs):
        raise AssertionError("cache hit should not re-run BSM1 rollout")

    monkeypatch.setattr(bsm1, "generate_bsm1_pretest_dataset", _boom)
    second = bsm1.generate_bsm1_pretest_stage_dataset(cfg, seed=0, workers=1)
    assert set(first) == set(second)
    for split in first:
        assert np.array_equal(first[split].states, second[split].states)


def test_ensemble_lstm_parallel_fit_matches_sequential() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("timesim")
    from scs.models.ensemble_lstm import EnsembleLSTMModel

    train = _synthetic_linear_batch()
    seq = EnsembleLSTMModel(random_state=0, n_members=3, hidden_dim=8, num_layers=1, epochs=3)
    par = EnsembleLSTMModel(random_state=0, n_members=3, hidden_dim=8, num_layers=1, epochs=3)

    seq.fit(train, workers=1)
    par.fit(train, workers=2)

    seq_pred = seq.predict_rollouts(train.states[:, 0], train.actions, train.disturbances)
    par_pred = par.predict_rollouts(train.states[:, 0], train.actions, train.disturbances)
    assert np.allclose(par_pred, seq_pred, rtol=1e-6, atol=1e-6)
    for m_seq, m_par in zip(seq.members, par.members):
        assert np.allclose(m_par.train_losses, m_seq.train_losses, rtol=1e-6, atol=1e-6)


def test_run_v3_stage_parallel_matches_sequential_across_seeds_and_ensembles(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("timesim")
    from scs.experiments.v3 import run_v3_stage

    cfg = _tiny_v3_ode_config("configs/v2/v2_event_targets.yaml")
    cfg_path = tmp_path / "tiny_stage_a.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    out_seq = tmp_path / "seq"
    out_par = tmp_path / "par"
    res_seq = run_v3_stage(cfg_path, out_seq, workers=1)
    res_par = run_v3_stage(cfg_path, out_par, workers=2)

    assert res_par["rows"] == res_seq["rows"]
    assert res_par["eligible_systems"] == res_seq["eligible_systems"]
    df_seq = pd.read_csv(res_seq["results"])
    df_par = pd.read_csv(res_par["results"])
    pd.testing.assert_frame_equal(df_par, df_seq, check_exact=False, rtol=1e-5, atol=1e-5)


def test_run_v3_stage_bsm1_rehearsal_parallel_matches_sequential(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("bsm2_python")
    pytest.importorskip("torch")
    pytest.importorskip("timesim")
    from scs.experiments.v3 import run_v3_stage
    from scs.systems import bsm1
    from tests.test_v3_bsm1_interface import _state_after_short_rollout

    state = _state_after_short_rollout()
    monkeypatch.setattr(bsm1, "_cached_start_states", lambda dry, starts: {int(start): state for start in starts})
    monkeypatch.setattr(bsm1, "_cached_rain_start_states", lambda dry, rain, starts: {int(start): state for start in starts})

    base = {
        "stage": "B-rehearsal",
        "label": "rehearsal",
        "protocol_frozen": False,
        "event_config": "configs/v3/bsm1_events.yaml",
        "check_threshold_share": False,
        "systems": ["bsm1"],
        "models": ["hold_last", "linear_narx", "lstm_world", "ensemble_lstm"],
        "learned_models": ["lstm_world", "ensemble_lstm"],
        "eligibility_split": "judge_calibration_id",
        "eligibility_baselines": ["hold_last", "linear_narx"],
        "primary_coverages": [0.25, 0.50],
        "descriptive_coverages": [0.75],
        "badness_targets": ["bad_rmse", "bad_event"],
        "rmse_threshold": {"bsm1": 0.5},
        "rmse_metric": "standardized_state",
        "event_guards": {"event_guard_no_invariant": ["risk_support_only", "risk_uncertainty_only", "risk_disagreement_only"]},
        "seeds": [0, 1],
        "bootstrap_samples": 20,
        "bootstrap_seed": 1,
        "interval_width_limit": 0.50,
        "min_test_scenarios": 8,
        "data": {
            "horizon": 4,
            "history_minutes": 2,
            "dt": 1.0 / 1440.0,
            "n_model_train": 6,
            "n_calibration_id": 2,
            "n_calibration_ood": 2,
            "n_test_per_split": 2,
            "uncertainty_samples": 5,
        },
        "model_runtime": {
            "ensemble_members": 5,
            "lstm_hidden": 8,
            "lstm_layers": 1,
            "lstm_epochs": 2,
            "early_stopping": False,
        },
    }
    seq_cfg = {**base, "data": {**base["data"], "cache_dir": str(tmp_path / "cache_seq")}}
    par_cfg = {**base, "data": {**base["data"], "cache_dir": str(tmp_path / "cache_par")}}
    seq_cfg_path = tmp_path / "bsm1_seq.yaml"
    par_cfg_path = tmp_path / "bsm1_par.yaml"
    seq_cfg_path.write_text(yaml.safe_dump(seq_cfg), encoding="utf-8")
    par_cfg_path.write_text(yaml.safe_dump(par_cfg), encoding="utf-8")

    res_seq = run_v3_stage(seq_cfg_path, tmp_path / "out_seq", workers=1)
    res_par = run_v3_stage(par_cfg_path, tmp_path / "out_par", workers=2)
    df_seq = pd.read_csv(res_seq["results"])
    df_par = pd.read_csv(res_par["results"])
    pd.testing.assert_frame_equal(df_par, df_seq, check_exact=False, rtol=1e-5, atol=1e-5)


def test_v3_run_cli_accepts_workers_flag(tmp_path: Path) -> None:
    help_proc = subprocess.run(
        [sys.executable, "scripts/v3_run.py", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--workers" in help_proc.stdout

    cfg = _tiny_v3_ode_config("configs/v2/v2_event_targets.yaml")
    cfg["models"] = ["hold_last", "linear_narx", "mlp_state_space", "ensemble_mlp"]
    cfg["learned_models"] = ["mlp_state_space", "ensemble_mlp"]
    cfg_path = tmp_path / "cli_stage.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    out_dir = tmp_path / "cli_out"
    proc = subprocess.run(
        [sys.executable, "scripts/v3_run.py", "--config", str(cfg_path), "--output", str(out_dir), "--workers", "2"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "results=" in proc.stdout
    assert (out_dir / "stage_a_results.csv").is_file()
