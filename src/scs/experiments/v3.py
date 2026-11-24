"""Stage runner for the v3 refusal benchmark.

Margins for Q1 and Q2 are against a seeded random ranking. They are not the v2
margin, which picks the lowest test-set false-accept rate among simple and
stronger baselines.
"""

from __future__ import annotations

import hashlib
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scs.experiments.registry import make_system
from scs.experiments.v2 import (
    SIGNAL_COLUMNS,
    SIMPLE_RISK_COLUMN,
    _make_v2_model,
    _split_role,
    generate_v2_dataset,
    load_event_config,
    split_overlap_report,
    trajectory_bad_event,
    trajectory_event_flags,
)
from scs.models.base import predict_rollouts_compat
from scs.signals import available_signals
from scs.validators.invariants import invariant_residual_score
from scs.validators.repair import repair_amount_score
from scs.validators.support import SupportDistance

PRIMARY_JUDGE = "calibration_selected_candidate_ranker"
RANDOM_JUDGE = "random_baseline"
ALLOWED_COVERAGE_RULES = frozenset({"both", "either", "pooled"})
ALLOWED_CONTROL_SPLIT_INFLUENTS = frozenset({"same_as_nominal_split"})


def load_v3_config(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    required = [
        "systems",
        "models",
        "learned_models",
        "eligibility_baselines",
        "seeds",
        "primary_coverages",
        "badness_targets",
        "rmse_threshold",
        "event_config",
        "data",
        "model_runtime",
        "bootstrap_samples",
        "bootstrap_seed",
        "min_test_scenarios",
        "interval_width_limit",
    ]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"missing v3 config keys: {missing}")
    if "event_guards" not in data and "event_guard_inputs" not in data:
        raise ValueError("config needs event_guards or event_guard_inputs")
    threshold = data["rmse_threshold"]
    if isinstance(threshold, dict):
        absent = [str(name) for name in data["systems"] if name not in threshold]
        if absent:
            raise ValueError(f"rmse_threshold has no value for systems: {absent}")
    members = int(data["model_runtime"].get("ensemble_members", 0))
    if members < 5:
        raise ValueError("v3 requires an ensemble of at least 5 members")
    return data


DEFAULT_DEGENERATE_SHARE = 0.01
# The practical thresholds of the yes rule in the protocol. Q3 has to beat the primary ranker
# and the best simple baseline by at least the absolute one, so it needs room above zero.
PRACTICAL_ABSOLUTE_REDUCTION = 0.05
PRACTICAL_RELATIVE_REDUCTION = 0.10
FULL_STATE_RISK_COLUMNS = ("risk_invariant_only", "risk_repair_only")


def event_guard_definitions(config: dict[str, Any]) -> dict[str, list[str]]:
    """Named event guards and their input columns.

    ``event_guards`` maps a guard name to its inputs. A config with only
    ``event_guard_inputs`` has one guard, named ``event_guard``.
    """
    if "event_guards" in config:
        return {str(name): [str(column) for column in columns] for name, columns in config["event_guards"].items()}
    return {"event_guard": [str(column) for column in config["event_guard_inputs"]]}


def rmse_threshold_for(config: dict[str, Any], system_id: str) -> float:
    """The ``bad_rmse`` threshold for one system. A single number applies to every system."""
    value = config["rmse_threshold"]
    return float(value[system_id]) if isinstance(value, dict) else float(value)


def rmse_metric_for(config: dict[str, Any], system_id: str) -> str:
    """Whether RMSE is computed in raw state units or standardized over active ``model_train`` channels."""
    metric = config.get("rmse_metric", "raw")
    value = str(metric[system_id]) if isinstance(metric, dict) else str(metric)
    if value not in {"raw", "standardized_state"}:
        raise ValueError(f"unsupported rmse_metric for {system_id}: {value}")
    return value


def fit_state_scale(states: np.ndarray, floor: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Non-constant state mask and per-channel training standard deviation from ``model_train`` only."""
    values = np.asarray(states, dtype=float)
    std = values.reshape(-1, values.shape[-1]).std(axis=0)
    active = std >= float(floor)
    if not np.any(active):
        raise RuntimeError("all state channels are constant on model_train")
    return active, std[active]


def scaled_scenario_rmse(
    predicted: np.ndarray,
    true: np.ndarray,
    state_scale: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Per-scenario trajectory RMSE, optionally standardized across active channels."""
    diff = np.asarray(predicted, dtype=float) - np.asarray(true, dtype=float)
    if state_scale is not None:
        active, std = state_scale
        diff = diff[..., active] / std
    return np.sqrt(np.mean(diff**2, axis=(1, 2)))


def _batch_disagreement(
    predictions: list[np.ndarray],
    state_scale: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    if len(predictions) < 2:
        return np.zeros(predictions[0].shape[0], dtype=float)
    distances = []
    for left_index, left in enumerate(predictions):
        for right in predictions[left_index + 1 :]:
            distances.append(scaled_scenario_rmse(left, right, state_scale))
    return np.mean(np.stack(distances, axis=0), axis=0)


def _support_scores(support: SupportDistance, actions: np.ndarray, disturbances: np.ndarray) -> np.ndarray:
    if support.mean_ is None or support.std_ is None:
        raise RuntimeError("support distance must be fit before scoring")
    features = np.concatenate([actions, disturbances], axis=-1)
    scaled = (features - support.mean_) / support.std_
    return np.mean(np.linalg.norm(scaled, axis=-1), axis=1)


def _rollout_and_uncertainty(
    model: Any,
    initial_states: np.ndarray,
    actions: np.ndarray,
    disturbances: np.ndarray,
    n_samples: int,
    rng: np.random.Generator | None = None,
    history: np.ndarray | None = None,
    project=None,
    state_scale: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean rollout and epistemic uncertainty spread without running ensemble members twice.

    Deep ensembles roll out each member once and reuse those predictions for both the
    mean trajectory and the spread. Deterministic single models without native epistemic
    sampling return NaN for uncertainty so judges treat it as a missing signal rather
    than injecting homoscedastic training-residual noise.
    """
    del rng
    n_rows = initial_states.shape[0]
    members = getattr(model, "members", None)
    if members:
        member_preds = [
            predict_rollouts_compat(member, initial_states, actions, disturbances, history=history, project=project)
            for member in members
        ]
        prediction = np.mean(np.stack(member_preds, axis=0), axis=0)
        if n_samples <= 0:
            return prediction, np.full(n_rows, np.nan, dtype=float)
        selected = [member_preds[index % len(member_preds)] for index in range(n_samples)]
        stacked = np.stack(selected, axis=0)
        if state_scale is not None:
            active, std = state_scale
            stacked = stacked[..., active] / std
        return prediction, np.mean(np.std(stacked, axis=0), axis=(1, 2))
    prediction = predict_rollouts_compat(model, initial_states, actions, disturbances, history=history, project=project)
    if getattr(model, "supports_epistemic_sampling", False) and n_samples > 0:
        samples = np.stack(
            [
                model.predict_rollout_samples(initial_states[index], actions[index], disturbances[index], n_samples)
                for index in range(n_rows)
            ],
            axis=1,
        )
        if project is not None:
            for step in range(actions.shape[1]):
                for sample_idx in range(samples.shape[0]):
                    samples[sample_idx, :, step + 1, :] = np.asarray(
                        project(step, samples[sample_idx, :, step + 1, :]), dtype=float
                    )
        if state_scale is not None:
            active, std = state_scale
            samples = samples[..., active] / std
        return prediction, np.mean(np.std(samples, axis=0), axis=(1, 2))
    return prediction, np.full(n_rows, np.nan, dtype=float)


def event_mismatch_per_limit(true_states: np.ndarray, predicted_states: np.ndarray, event_config: dict[str, Any]) -> bool:
    """BSM1 target: bad when any of the five effluent-limit bits differs, not only the combined bit.

    On the dry calibration rows the combined any-limit bit never disagrees, because the open-loop
    plant is nearly always over the ammonia limit. The limits are separate constraints, so each
    one is its own bit. Any non-finite predicted trajectory counts as a mismatch.
    """
    if not np.isfinite(predicted_states).all():
        return True
    true_flags = trajectory_event_flags("bsm1", true_states, event_config)
    predicted_flags = trajectory_event_flags("bsm1", predicted_states, event_config)
    return any(true_flags[name] != predicted_flags[name] for name in true_flags)


def disagreement_cohort(system_id: str, eligibility: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    """Models that contribute to ``disagreement_score``, selected strictly from calibration eligibility.

    Uses calibration-eligible learned models when at least two exist; otherwise falls back to
    configured learned models (or non-``hold_last`` models), excluding ``hold_last`` and
    diverging ineligible models wherever possible.
    """
    eligible = [
        str(row.model_id)
        for row in eligibility.itertuples(index=False)
        if str(row.system_id) == str(system_id) and bool(row.eligible)
    ]
    if len(eligible) >= 2:
        return eligible
    learned = [str(model_id) for model_id in config.get("learned_models", [])]
    if len(learned) >= 2:
        return learned
    non_trivial = [str(model_id) for model_id in config.get("models", []) if str(model_id) != "hold_last"]
    if len(non_trivial) >= 2:
        return non_trivial
    return [str(model_id) for model_id in config.get("models", [])]


def _rollout_batch(
    system_id: str,
    seed: int,
    split: str,
    batch: Any,
    models: dict[str, Any],
    support: SupportDistance,
    event_config: dict[str, Any],
    dt: float,
    n_uncertainty: int,
    state_scale: tuple[np.ndarray, np.ndarray] | None = None,
    disagreement_models: list[str] | None = None,
    keep_predictions: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    system = make_system(system_id)
    initial = batch.states[:, 0, :]
    project = None
    if system_id == "bsm1":
        from scs.systems.bsm1 import hydraulic_projection

        project = hydraulic_projection(batch.disturbances)
    predictions: dict[str, np.ndarray] = {}
    uncertainty: dict[str, np.ndarray] = {}
    for model_id, model in models.items():
        rng = np.random.default_rng(10_000 + seed * 100 + sum(ord(char) for char in model_id))
        pred, unc = _rollout_and_uncertainty(
            model,
            initial,
            batch.actions,
            batch.disturbances,
            n_uncertainty,
            rng=rng,
            history=batch.measured_history,
            project=project,
            state_scale=state_scale,
        )
        predictions[model_id] = pred
        uncertainty[model_id] = unc
    cohort = [model_id for model_id in (disagreement_models or list(models)) if model_id in predictions]
    if len(cohort) < 2:
        cohort = list(models)
    disagreement = _batch_disagreement([predictions[model_id] for model_id in cohort], state_scale=state_scale)
    support_scores = _support_scores(support, batch.actions, batch.disturbances)
    true_events = [
        trajectory_bad_event(system_id, batch.states[index], event_config) for index in range(batch.n_trajectories)
    ]
    rmses = {
        model_id: scaled_scenario_rmse(predicted, batch.states, state_scale=state_scale)
        for model_id, predicted in predictions.items()
    }
    signals_here = available_signals(system_id)
    rows: list[dict[str, Any]] = []
    role = _split_role(split)
    for index in range(batch.n_trajectories):
        true_states = batch.states[index]
        for model_id, predicted in predictions.items():
            pred = predicted[index]
            finite_pred = bool(np.isfinite(pred).all())
            pred_event = trajectory_bad_event(system_id, pred, event_config) if finite_pred else False
            rows.append(
                {
                    "system_id": system_id,
                    "seed": seed,
                    "split": split,
                    "role": role,
                    "scenario_type": batch.scenario_type[index],
                    "scenario_id": f"{system_id}_seed{seed}_{split}_{index:04d}",
                    "model_id": model_id,
                    "rmse": float(rmses[model_id][index]),
                    "true_event": bool(true_events[index]),
                    "pred_event": bool(pred_event),
                    "event_mismatch": (
                        event_mismatch_per_limit(true_states, pred, event_config)
                        if system_id == "bsm1"
                        else (True if not finite_pred else bool(true_events[index] != pred_event))
                    ),
                    "support_distance": float(support_scores[index]),
                    "uncertainty_score": float(uncertainty[model_id][index]),
                    "disagreement_score": float(disagreement[index]),
                    "invariant_residual": (
                        invariant_residual_score(system, pred, batch.actions[index], batch.disturbances[index], dt)
                        if "invariant_residual" in signals_here
                        else float("nan")
                    ),
                    "repair_amount": (
                        repair_amount_score(system, pred) if "repair_amount" in signals_here else float("nan")
                    ),
                }
            )
    if not keep_predictions:
        return rows, {}
    # Store float32 predictions for calibration disagreement recomputation to keep multi-seed RAM low;
    # drop hold_last whenever at least two other models exist since hold_last never enters the cohort.
    retained_ids = [model_id for model_id in predictions if model_id != "hold_last"]
    if len(retained_ids) < 2:
        retained_ids = list(predictions)
    kept = {model_id: predictions[model_id].astype(np.float32, copy=False) for model_id in retained_ids}
    return rows, kept


DEFAULT_ELIGIBILITY_SPLIT = "judge_calibration_id"


def eligibility_split(config: dict[str, Any]) -> str:
    """The in-distribution split that decides eligibility. Never a judge-test split.

    Eligibility has to be known before the confirmatory test split exists, and a
    model must not be kept or dropped on test outcomes.
    """
    split = str(config.get("eligibility_split", DEFAULT_ELIGIBILITY_SPLIT))
    if _split_role(split) == "judge_test":
        raise ValueError(f"eligibility cannot use the judge-test split {split}")
    return split


def eligibility_table(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """A learned model is eligible when its mean in-distribution RMSE is strictly lower than both baselines."""
    split = eligibility_split(config)
    in_distribution = frame[frame["split"] == split]
    if in_distribution.empty:
        raise RuntimeError(f"no rows on the eligibility split {split}")
    means = (
        in_distribution.groupby(["system_id", "model_id"], as_index=False)["rmse"]
        .mean()
        .rename(columns={"rmse": "id_rmse"})
    )
    rows = []
    for system_id, group in means.groupby("system_id", sort=False):
        baselines = {
            model_id: float(group.loc[group["model_id"] == model_id, "id_rmse"].iloc[0])
            for model_id in config["eligibility_baselines"]
        }
        for row in group.itertuples(index=False):
            learned = row.model_id in config["learned_models"]
            id_val = float(row.id_rmse)
            beats = learned and math.isfinite(id_val) and all(id_val < baseline for baseline in baselines.values())
            rows.append(
                {
                    "system_id": system_id,
                    "model_id": row.model_id,
                    "id_rmse": id_val,
                    "eligible": bool(beats),
                    **{f"baseline_{name}_id_rmse": value for name, value in baselines.items()},
                }
            )
    return pd.DataFrame(rows)


def threshold_share_check(frame: pd.DataFrame, eligibility: pd.DataFrame, config: dict[str, Any]) -> None:
    """Refuse a bad_rmse threshold that marks almost none or almost all calibration scenarios as bad.

    Reads calibration rows of eligible learned models only. A threshold that does that makes
    the margins on that target fixed by the labels. This catches it before test rows are scored.
    """
    edge = float(config.get("degenerate_share", DEFAULT_DEGENERATE_SHARE))
    chosen = {(row.system_id, row.model_id) for row in eligibility.itertuples(index=False) if row.eligible}
    for system_id in config["systems"]:
        rows = frame[(frame["system_id"] == system_id) & (frame["role"] == "judge_calibration")]
        rows = rows[[(system_id, model_id) in chosen for model_id in rows["model_id"]]]
        if rows.empty:
            continue
        threshold = rmse_threshold_for(config, str(system_id))
        rmse_vals = rows["rmse"].to_numpy(dtype=float)
        share = float(((~np.isfinite(rmse_vals)) | (rmse_vals > threshold)).mean())
        if share < edge or share > 1.0 - edge:
            raise RuntimeError(
                f"the bad_rmse threshold {threshold:g} marks {share:.1%} of {system_id} calibration scenarios "
                f"for eligible models as bad, outside {edge:.0%} to {1 - edge:.0%}. Its margins would be set by the labels."
            )


def _relative_margin(absolute: float, baseline_far: float) -> float:
    if baseline_far > 0.0 and math.isfinite(baseline_far) and math.isfinite(absolute):
        return float(absolute / baseline_far)
    return float("nan")


def _finite_signal_names(frame: pd.DataFrame) -> list[str]:
    names = []
    for name, column in SIMPLE_RISK_COLUMN.items():
        values = frame[column].to_numpy(dtype=float)
        if np.isfinite(values).all():
            names.append(name)
    return names


def _sanitize_risk_scores(scores: np.ndarray) -> np.ndarray:
    """Map any non-finite test risk score (e.g. from a diverged rollout) to +inf (maximum risk)."""
    values = np.asarray(scores, dtype=float)
    return np.where(np.isfinite(values), values, np.inf)


def _tie_aware_risk_curve(labels: np.ndarray, risk_scores: np.ndarray, coverages: list[float]) -> pd.DataFrame:
    """Exact expected risk-coverage curve under uniform random tie-breaking at each coverage cutoff.

    Permutation-invariant with respect to row order, so constant or quantized-rank scores
    cannot exploit split ordering (such as ID rows preceding OOD rows). Non-finite risk scores
    are treated as +inf (maximum risk, refused first).
    """
    labels = np.asarray(labels, dtype=bool)
    scores = _sanitize_risk_scores(risk_scores)
    n = len(labels)
    if n == 0:
        raise ValueError("at least one scenario required")
    sorted_scores = np.sort(scores)
    rows = []
    for coverage in coverages:
        count = min(max(int(math.ceil(float(coverage) * n)), 1), n)
        cutoff = sorted_scores[count - 1]
        below = scores < cutoff
        equal = scores == cutoff
        n_below = int(np.sum(below))
        bad_below = float(np.sum(labels[below]))
        remaining = count - n_below
        n_equal = int(np.sum(equal))
        bad_equal = float(np.sum(labels[equal]))
        expected_bad = bad_below + (remaining * (bad_equal / n_equal) if n_equal > 0 and remaining > 0 else 0.0)
        rows.append(
            {
                "coverage": float(coverage),
                "accepted_count": count,
                "false_accept_count": expected_bad,
                "false_accept_rate": float(expected_bad / count),
            }
        )
    return pd.DataFrame(rows)


def _tie_aware_low_coverage_far(labels: np.ndarray, score: np.ndarray, coverages: list[float]) -> float:
    return float(_tie_aware_risk_curve(labels, score, coverages)["false_accept_rate"].mean())


def _signal_orientation_for(calibration: pd.DataFrame, column: str) -> int:
    labels = calibration["bad_label"].to_numpy(dtype=bool)
    values = calibration[column].to_numpy(dtype=float)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return 1
    return 1 if float(values[labels].mean()) >= float(values[~labels].mean()) else -1


def _oriented_values(raw: np.ndarray, orientation: int) -> np.ndarray:
    values = np.asarray(raw, dtype=float)
    return np.where(np.isfinite(values), orientation * values, np.inf)


def _rank_mean_columns(
    calibration: pd.DataFrame,
    table: pd.DataFrame,
    columns: list[str],
    orientations: dict[str, int],
) -> np.ndarray:
    parts = []
    for column in columns:
        reference = np.sort(orientations[column] * calibration[column].to_numpy(dtype=float))
        raw = table[column].to_numpy(dtype=float)
        ranks = np.full(len(raw), np.inf, dtype=float)
        finite = np.isfinite(raw)
        if finite.any():
            ranks[finite] = np.searchsorted(reference, orientations[column] * raw[finite], side="right") / max(
                len(reference), 1
            )
        parts.append(ranks)
    return np.mean(np.vstack(parts), axis=0)


def _natural_sum_columns(
    calibration: pd.DataFrame,
    table: pd.DataFrame,
    columns: list[str],
    orientations: dict[str, int],
) -> np.ndarray:
    parts = []
    for column in columns:
        cal_vals = orientations[column] * calibration[column].to_numpy(dtype=float)
        lo = float(np.min(cal_vals))
        denom = max(float(np.max(cal_vals) - lo), 1e-12)
        raw = table[column].to_numpy(dtype=float)
        scaled = np.full(len(raw), np.inf, dtype=float)
        finite = np.isfinite(raw)
        if finite.any():
            scaled[finite] = (orientations[column] * raw[finite] - lo) / denom
        parts.append(scaled)
    return np.mean(np.vstack(parts), axis=0)


def _event_guarded_score_safe(
    calibration: pd.DataFrame,
    table: pd.DataFrame,
    input_columns: list[str],
) -> np.ndarray:
    """Maximum calibration-percentile rank across ``input_columns``; non-finite test values get +inf."""
    if calibration.empty:
        raise ValueError("calibration rows are required")
    ranks = []
    for column in input_columns:
        reference = np.sort(calibration[column].to_numpy(dtype=float))
        raw = table[column].to_numpy(dtype=float)
        col_rank = np.full(len(raw), np.inf, dtype=float)
        finite = np.isfinite(raw)
        if finite.any():
            col_rank[finite] = np.searchsorted(reference, raw[finite], side="right") / max(len(reference), 1)
        ranks.append(col_rank)
    return np.maximum.reduce(ranks)


def _calibration_auc(labels: np.ndarray, score: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return 0.5
    return float(roc_auc_score(labels.astype(int), np.asarray(score, dtype=float)))


def _fit_calibrated_ranker(
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    primary_coverages: list[float],
) -> np.ndarray:
    """Calibration-selected candidate ranker fit per model over signals finite on calibration.

    Selects among the four v2 candidates (``best_signal``, ``rank_mean``, ``natural_sum``,
    ``logistic``) strictly on calibration rows (never inspecting test finiteness or labels).
    Any non-finite test signal value is mapped to +inf (maximum risk).
    """
    available = [column for column in SIGNAL_COLUMNS if np.isfinite(calibration[column].to_numpy(dtype=float)).all()]
    if not available:
        raise RuntimeError("no finite refusal signal")
    labels = calibration["bad_label"].to_numpy(dtype=bool)
    orientations = {column: _signal_orientation_for(calibration, column) for column in available}
    single_candidates = []
    for column in available:
        cal_score = orientations[column] * calibration[column].to_numpy(dtype=float)
        single_candidates.append(
            (
                _tie_aware_low_coverage_far(labels, cal_score, primary_coverages),
                -_calibration_auc(labels, cal_score),
                column,
            )
        )
    best_column = sorted(single_candidates)[0][2]
    cal_best = orientations[best_column] * calibration[best_column].to_numpy(dtype=float)
    test_best = _oriented_values(test[best_column].to_numpy(dtype=float), orientations[best_column])
    cal_rank_mean = _rank_mean_columns(calibration, calibration, available, orientations)
    test_rank_mean = _rank_mean_columns(calibration, test, available, orientations)
    cal_natural = _natural_sum_columns(calibration, calibration, available, orientations)
    test_natural = _natural_sum_columns(calibration, test, available, orientations)
    if 0 < int(labels.sum()) < len(labels):
        logistic = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, class_weight="balanced"))
        logistic.fit(calibration[available], labels.astype(int))
        cal_logistic = logistic.predict_proba(calibration[available])[:, 1]
        test_logistic = np.full(len(test), np.inf, dtype=float)
        finite_rows = np.isfinite(test[available].to_numpy(dtype=float)).all(axis=1)
        if finite_rows.any():
            test_logistic[finite_rows] = logistic.predict_proba(test.loc[finite_rows, available])[:, 1]
    else:
        cal_logistic = cal_rank_mean
        test_logistic = test_rank_mean
    cal_candidates = {
        "best_signal": cal_best,
        "rank_mean": cal_rank_mean,
        "natural_sum": cal_natural,
        "logistic": cal_logistic,
    }
    test_candidates = {
        "best_signal": test_best,
        "rank_mean": test_rank_mean,
        "natural_sum": test_natural,
        "logistic": test_logistic,
    }
    selected = min(
        cal_candidates,
        key=lambda name: (
            _tie_aware_low_coverage_far(labels, cal_candidates[name], primary_coverages),
            -_calibration_auc(labels, cal_candidates[name]),
            name,
        ),
    )
    return test_candidates[selected]


def _best_simple_baseline(calibration: pd.DataFrame, coverages: list[float]) -> str:
    labels = calibration["bad_label"].to_numpy(dtype=bool)
    best_name = ""
    best_key: tuple[float, float, str] | None = None
    finite_names = _finite_signal_names(calibration)
    if not finite_names:
        raise RuntimeError("no finite simple baseline")
    for name in finite_names:
        column = SIMPLE_RISK_COLUMN[name]
        values = calibration[column].to_numpy(dtype=float)
        far = _tie_aware_low_coverage_far(labels, values, coverages)
        key = (far, -_calibration_auc(labels, values), name)
        if best_key is None or key < best_key:
            best_key = key
            best_name = name
    if not best_name:
        raise RuntimeError("no simple baseline could be selected")
    return best_name


def _assign_random(test: pd.DataFrame, seed: int) -> np.ndarray:
    scores = np.empty(len(test), dtype=float)
    positions = pd.Series(np.arange(len(test)), index=test.index)
    for model_index, (_model_id, group) in enumerate(test.groupby("model_id", sort=True)):
        rng = np.random.default_rng(50_000 + int(seed) * 100 + model_index)
        scores[positions.loc[group.index].to_numpy()] = rng.random(len(group))
    return scores


def result_rows_for_frame(frame: pd.DataFrame, config: dict[str, Any], eligibility: pd.DataFrame) -> pd.DataFrame:
    coverages = [float(value) for value in config["primary_coverages"]]
    # Descriptive coverages are scored for every judge, but nothing is selected or decided on them.
    descriptive = [float(value) for value in config.get("descriptive_coverages", [])]
    curve_coverages = coverages + [value for value in descriptive if not any(np.isclose(value, c) for c in coverages)]
    eligible = {
        (row.system_id, row.model_id): bool(row.eligible) for row in eligibility.itertuples(index=False)
    }
    pieces = []
    for (system_id, seed), group in frame.groupby(["system_id", "seed"], sort=True):
        threshold = rmse_threshold_for(config, str(system_id))
        for target in config["badness_targets"]:
            labelled = group.copy()
            if target == "bad_rmse":
                rmse_vals = labelled["rmse"].to_numpy(dtype=float)
                labelled["bad_label"] = (~np.isfinite(rmse_vals)) | (rmse_vals > threshold)
            elif target == "bad_event":
                labelled["bad_label"] = labelled["event_mismatch"].astype(bool)
            else:
                raise ValueError(f"unsupported badness target: {target}")
            for column, signal in SIMPLE_RISK_COLUMN.items():
                labelled[f"risk_{column}"] = labelled[signal]
            calibration = labelled[labelled["role"] == "judge_calibration"].copy()
            test = labelled[labelled["role"] == "judge_test"].copy()
            test["risk_random_baseline"] = _assign_random(test, int(seed))
            for model_id, model_test in test.groupby("model_id", sort=True):
                model_calibration = calibration[calibration["model_id"] == model_id]
                simple_name = _best_simple_baseline(model_calibration, coverages)
                primary_scores = _fit_calibrated_ranker(model_calibration, model_test, coverages)
                judge_scores = {
                    name: _sanitize_risk_scores(model_test[f"risk_{name}"].to_numpy(dtype=float))
                    for name in _finite_signal_names(model_calibration)
                }
                judge_scores[RANDOM_JUDGE] = model_test["risk_random_baseline"].to_numpy(dtype=float)
                judge_scores[PRIMARY_JUDGE] = primary_scores
                # A guard is defined by all of its inputs on calibration. When one is missing on
                # calibration it is not computed on a smaller set, because that would be a different guard.
                for guard_id, inputs in event_guard_definitions(config).items():
                    ready = all(
                        np.isfinite(model_calibration[column].to_numpy(dtype=float)).all() for column in inputs
                    )
                    if ready:
                        judge_scores[guard_id] = _event_guarded_score_safe(model_calibration, model_test, list(inputs))
                labels = model_test["bad_label"].to_numpy(dtype=bool)
                random_curve = _tie_aware_risk_curve(labels, judge_scores[RANDOM_JUDGE], curve_coverages)
                primary_curve = _tie_aware_risk_curve(labels, judge_scores[PRIMARY_JUDGE], curve_coverages)
                simple_curve = _tie_aware_risk_curve(labels, judge_scores[simple_name], curve_coverages)
                primary_far = {float(row.coverage): float(row.false_accept_rate) for row in primary_curve.itertuples(index=False)}
                simple_far = {float(row.coverage): float(row.false_accept_rate) for row in simple_curve.itertuples(index=False)}
                for judge_id, scores in judge_scores.items():
                    curve = _tie_aware_risk_curve(labels, scores, curve_coverages)
                    for row, random_row in zip(curve.itertuples(index=False), random_curve.itertuples(index=False), strict=True):
                        absolute = float(random_row.false_accept_rate - row.false_accept_rate)
                        pieces.append(
                            {
                                "system_id": system_id,
                                "seed": int(seed),
                                "model_id": model_id,
                                "eligible": bool(eligible.get((system_id, model_id), False)),
                                "badness_target": target,
                                "coverage": float(row.coverage),
                                "coverage_role": (
                                    "primary" if any(np.isclose(float(row.coverage), c) for c in coverages) else "descriptive"
                                ),
                                "judge_id": judge_id,
                                "false_accept_rate": float(row.false_accept_rate),
                                "random_far": float(random_row.false_accept_rate),
                                "absolute_margin": absolute,
                                "relative_margin": _relative_margin(absolute, float(random_row.false_accept_rate)),
                                "primary_far": primary_far[float(row.coverage)],
                                "calibration_simple_far": simple_far[float(row.coverage)],
                                "bad_share": float(np.mean(labels)),
                                "accepted_count": int(row.accepted_count),
                                "n_scenarios": int(len(model_test)),
                                "calibration_simple_baseline": simple_name,
                                "rmse_threshold": threshold,
                            }
                        )
    return pd.DataFrame(pieces)


def bootstrap_interval(values: np.ndarray, rng: np.random.Generator, n_samples: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float("nan"), float("nan")
    samples = np.empty(n_samples, dtype=float)
    for index in range(n_samples):
        samples[index] = float(np.mean(rng.choice(values, size=len(values), replace=True)))
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def interval_table(results: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Bootstrap the seed-level mean margin of eligible models, separately at each primary coverage.

    Relative reduction uses ratio of means (``mean_margin / mean_random_far``), matching the Q3
    guard table so Q1, Q2, and Q3 share one definition and zero-FAR seeds are never silently dropped.
    A row is degenerate when almost none or almost all test scenarios are bad.
    """
    primary = results[(results["judge_id"] == PRIMARY_JUDGE) & (results["eligible"])].copy()
    if "coverage_role" in primary.columns:
        primary = primary[primary["coverage_role"] == "primary"]
    rng = np.random.default_rng(int(config["bootstrap_seed"]))
    rows = []
    if primary.empty:
        return pd.DataFrame(rows)
    agg_spec: dict[str, tuple[str, str]] = {
        "absolute_margin": ("absolute_margin", "mean"),
        "bad_share": ("bad_share", "mean"),
    }
    if "random_far" in primary.columns:
        agg_spec["random_far"] = ("random_far", "mean")
    grouped = primary.groupby(["system_id", "badness_target", "coverage", "seed"], as_index=False).agg(**agg_spec)
    limit = float(config["interval_width_limit"])
    n_samples = int(config["bootstrap_samples"])
    edge = float(config.get("degenerate_share", DEFAULT_DEGENERATE_SHARE))
    per_model = (
        primary.groupby(["system_id", "badness_target", "coverage", "model_id"], as_index=False)["bad_share"].mean()
        if "model_id" in primary.columns
        else None
    )
    for (system_id, target, coverage), group in grouped.groupby(["system_id", "badness_target", "coverage"], sort=True):
        low, high = bootstrap_interval(group["absolute_margin"].to_numpy(dtype=float), rng, n_samples)
        width = high - low if math.isfinite(low) and math.isfinite(high) else float("nan")
        share = float(group["bad_share"].mean())
        mean_margin = float(group["absolute_margin"].mean())
        n_models, thin_models = 0, 0
        if per_model is not None:
            models = per_model[
                (per_model["system_id"] == system_id)
                & (per_model["badness_target"] == target)
                & np.isclose(per_model["coverage"], coverage)
            ]
            n_models = int(len(models))
            thin_models = int(((models["bad_share"] < edge) | (models["bad_share"] > 1.0 - edge)).sum())
        rel_mean = (
            _relative_margin(mean_margin, float(group["random_far"].mean()))
            if "random_far" in group.columns
            else float("nan")
        )
        rows.append(
            {
                "system_id": system_id,
                "badness_target": target,
                "coverage": float(coverage),
                "n_seeds": int(group["seed"].nunique()),
                "bad_share": share,
                "degenerate": bool(share < edge or share > 1.0 - edge),
                "n_models": n_models,
                "thin_models": thin_models,
                "mean_margin": mean_margin,
                "mean_relative_margin": rel_mean,
                "bootstrap_low": low,
                "bootstrap_high": high,
                "interval_width": width,
                "narrower_than_limit": bool(math.isfinite(width) and width < limit),
            }
        )
    return pd.DataFrame(rows)


def precision_check(intervals: pd.DataFrame) -> tuple[bool, int]:
    """Whether every informative 5% interval is narrow, and how many 5% rows were degenerate.

    Degenerate rows are named, not counted as passes. The check also needs at
    least one informative row on every system that has an interval.
    """
    if intervals.empty:
        return False, 0
    at_five = intervals[np.isclose(intervals["coverage"], 0.05)]
    informative = at_five[~at_five["degenerate"]]
    covered = set(informative["system_id"]) == set(at_five["system_id"])
    passed = bool(covered and not informative.empty and informative["narrower_than_limit"].all())
    return passed, int(at_five["degenerate"].sum())


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_none_"
    shown = frame[columns]
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in shown.itertuples(index=False):
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append("" if math.isnan(value) else f"{value:.4f}")
            else:
                cells.append(str(value))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, rule, *body])


def _guard_lines(results: pd.DataFrame, config: dict[str, Any] | None = None) -> str:
    """Each event guard against the primary ranker and the calibration-chosen simple baseline.

    Reports every primary coverage (both 5% and 10% when present), relative reductions
    (ratio of means), and seed-level bootstrap CIs for Q3. Only eligible models enter the averages.
    """
    systems = sorted(results["system_id"].unique())
    primary_rows = results[results["coverage_role"] == "primary"] if "coverage_role" in results.columns else results
    discovered = {str(name) for name in primary_rows["judge_id"].unique() if str(name).startswith("event_guard")}
    if config and ("event_guards" in config or "event_guard_inputs" in config):
        discovered.update(event_guard_definitions(config))
    guard_ids = sorted(discovered)
    if not guard_ids:
        return "_none_"
    eligible_rows = primary_rows[primary_rows["eligible"]]
    n_samples = int(config["bootstrap_samples"]) if config and "bootstrap_samples" in config else 500
    seed_val = int(config["bootstrap_seed"]) if config and "bootstrap_seed" in config else 2025
    rng = np.random.default_rng(seed_val + 17)
    tables, notes = [], []
    for guard_id in guard_ids:
        guard = eligible_rows[eligible_rows["judge_id"] == guard_id].copy()
        has_rows = set(primary_rows.loc[primary_rows["judge_id"] == guard_id, "system_id"])
        no_guard = [name for name in systems if name not in has_rows]
        if no_guard:
            notes.append(f"`{guard_id}` is not defined on: {', '.join(no_guard)}. A required input is missing there.")
        no_eligible = [name for name in systems if name in has_rows and name not in set(guard["system_id"])]
        if no_eligible:
            notes.append(f"`{guard_id}` has no eligible model, so no row, on: {', '.join(no_eligible)}.")
        if guard.empty:
            continue
        guard["margin_vs_primary"] = guard["primary_far"] - guard["false_accept_rate"]
        guard["margin_vs_calibration_simple"] = guard["calibration_simple_far"] - guard["false_accept_rate"]
        summary_rows = []
        for (system_id, target, coverage), group in guard.groupby(
            ["system_id", "badness_target", "coverage"], sort=True
        ):
            guard_far = float(group["false_accept_rate"].mean())
            primary_far = float(group["primary_far"].mean())
            simple_far = float(group["calibration_simple_far"].mean())
            margin_primary = float(group["margin_vs_primary"].mean())
            margin_simple = float(group["margin_vs_calibration_simple"].mean())
            ci_low_p = ci_high_p = ci_low_s = ci_high_s = float("nan")
            if "seed" in group.columns and int(group["seed"].nunique()) >= 2:
                by_seed = group.groupby("seed", as_index=False).agg(
                    margin_vs_primary=("margin_vs_primary", "mean"),
                    margin_vs_calibration_simple=("margin_vs_calibration_simple", "mean"),
                )
                ci_low_p, ci_high_p = bootstrap_interval(
                    by_seed["margin_vs_primary"].to_numpy(dtype=float), rng, n_samples
                )
                ci_low_s, ci_high_s = bootstrap_interval(
                    by_seed["margin_vs_calibration_simple"].to_numpy(dtype=float), rng, n_samples
                )
            summary_rows.append(
                {
                    "guard": guard_id,
                    "system_id": system_id,
                    "badness_target": target,
                    "coverage": float(coverage),
                    "n_models": int(group["model_id"].nunique()),
                    "event_guard_far": guard_far,
                    "primary_far": primary_far,
                    "calibration_simple_far": simple_far,
                    "margin_vs_primary": margin_primary,
                    "rel_vs_primary": _relative_margin(margin_primary, primary_far),
                    "ci_low_vs_primary": ci_low_p,
                    "ci_high_vs_primary": ci_high_p,
                    "margin_vs_calibration_simple": margin_simple,
                    "rel_vs_simple": _relative_margin(margin_simple, simple_far),
                    "ci_low_vs_simple": ci_low_s,
                    "ci_high_vs_simple": ci_high_s,
                    "room_for_q3": bool(
                        primary_far >= PRACTICAL_ABSOLUTE_REDUCTION
                        and simple_far >= PRACTICAL_ABSOLUTE_REDUCTION
                    ),
                }
            )
        tables.append(pd.DataFrame(summary_rows))
    if not tables:
        return "\n\n".join(["_none_", *notes])
    table = _markdown_table(
        pd.concat(tables, ignore_index=True),
        [
            "guard",
            "system_id",
            "badness_target",
            "coverage",
            "n_models",
            "event_guard_far",
            "primary_far",
            "calibration_simple_far",
            "margin_vs_primary",
            "rel_vs_primary",
            "ci_low_vs_primary",
            "ci_high_vs_primary",
            "margin_vs_calibration_simple",
            "rel_vs_simple",
            "ci_low_vs_simple",
            "ci_high_vs_simple",
            "room_for_q3",
        ],
    )
    notes.append(
        f"`room_for_q3` is true only when both the primary ranker and the calibration-chosen simple baseline have a "
        f"false-accept rate of at least {PRACTICAL_ABSOLUTE_REDUCTION}. The yes rule asks for an absolute improvement of "
        f"{PRACTICAL_ABSOLUTE_REDUCTION} over both, and a false-accept rate cannot fall below zero, so where this is false Q3 "
        "cannot be yes whatever the guard does."
    )
    return "\n\n".join([table, *notes])


def _descriptive_lines(results: pd.DataFrame) -> str:
    """False-accept rates at the descriptive coverages. Not part of any decision rule."""
    if "coverage_role" not in results.columns:
        return "_none_"
    rows = results[(results["coverage_role"] == "descriptive") & results["eligible"]]
    if rows.empty:
        return "_none_"
    wide = rows.pivot_table(
        index=["system_id", "badness_target", "coverage"], columns="judge_id", values="false_accept_rate", aggfunc="mean"
    )
    primary_rows = rows[rows["judge_id"] == PRIMARY_JUDGE]
    simple = primary_rows.groupby(["system_id", "badness_target", "coverage"])["calibration_simple_far"].mean()
    wide["best_simple_on_calibration"] = simple
    wide["bad_share"] = primary_rows.groupby(["system_id", "badness_target", "coverage"])["bad_share"].mean()
    order = [
        "random_baseline",
        PRIMARY_JUDGE,
        "best_simple_on_calibration",
        "support_only",
        "uncertainty_only",
        "disagreement_only",
        "invariant_only",
        "repair_only",
        *sorted(column for column in wide.columns if str(column).startswith("event_guard")),
        "bad_share",
    ]
    wide = wide[[column for column in order if column in wide.columns]].reset_index()
    wide = wide.rename(columns={PRIMARY_JUDGE: "primary", "random_baseline": "random"})
    return _markdown_table(wide, list(wide.columns))


PROTOCOL_PATH = Path("docs/v3/protocol.md")


def protocol_is_frozen(path: str | Path = PROTOCOL_PATH) -> bool:
    """True only when the protocol status line says frozen.

    The draft says "not frozen", and that line does not count.
    """
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("Status:"):
            return line.startswith("Status: frozen") and "not frozen" not in line.lower()
    return False


def _tracked_working_tree_dirty() -> bool:
    """True when tracked repository files have uncommitted modifications."""
    try:
        output = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True)
        return bool(output.strip())
    except Exception:
        return True


def run_metadata(protocol_path: str | Path = PROTOCOL_PATH) -> dict[str, Any]:
    """Commit hash, working-tree cleanliness, and protocol digest written into results and reports."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    proto = Path(protocol_path)
    digest = hashlib.sha256(proto.read_bytes()).hexdigest() if proto.is_file() else ""
    return {
        "git_commit": commit,
        "git_dirty": _tracked_working_tree_dirty(),
        "protocol_sha256": digest,
        "protocol_frozen": protocol_is_frozen(proto) if proto.is_file() else False,
    }


def _write_report(
    path: Path,
    eligibility: pd.DataFrame,
    intervals: pd.DataFrame,
    results: pd.DataFrame,
    config: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> None:
    meta = metadata or run_metadata()
    learned = eligibility[eligibility["model_id"].isin(config["learned_models"])]
    systems_passed = sorted(learned.loc[learned["eligible"], "system_id"].unique())
    precision_ok, degenerate_rows = precision_check(intervals)
    runtime_text = ", ".join(
        f"{key} {value}" for key, value in sorted(config["model_runtime"].items()) if key != "ensemble_members"
    )
    thresholds = ", ".join(f"{name} {rmse_threshold_for(config, str(name)):g}" for name in config["systems"])
    metrics = ", ".join(f"{name} {rmse_metric_for(config, str(name))}" for name in config["systems"])
    stage = str(config.get("stage", "A"))
    label = str(config.get("label", "exploratory"))
    is_confirmatory = stage == "B"
    reading = (
        "This report lists the numbers the decision rules use. The reading of Q1, Q2, and Q3 is written by hand."
        if is_confirmatory
        else "This report does not answer Q1, Q2, or Q3."
    )
    baseline_names = [str(name) for name in config.get("eligibility_baselines", ["hold_last", "linear_narx"])]
    baseline_cols = [
        f"baseline_{name}_id_rmse"
        for name in baseline_names
        if f"baseline_{name}_id_rmse" in eligibility.columns
    ]
    eligibility_cols = ["system_id", "model_id", "id_rmse", *baseline_cols, "eligible"]
    eligibility_note = (
        f"Working rule for this run: mean RMSE on `{eligibility_split(config)}` is strictly below "
        f"{' and '.join(f'`{name}`' for name in baseline_names)} on that system. Eligibility does not read judge-test rows. "
        f"Learned models passing: {len(systems_passed)} system(s) ({', '.join(systems_passed) if len(systems_passed) else 'none'})."
    )
    if stage == "A":
        eligibility_note += " The plan's Stage A check wants at least two systems."
    settings_note = (
        "Frozen confirmatory settings from the stage config."
        if is_confirmatory
        else "These numbers are the working settings in the config. A freeze is a later commit."
    )
    guard_intro = (
        "Confirmatory Q3 numbers across all primary coverages. "
        if is_confirmatory
        else "Exploratory only. "
    )
    guard_intro += (
        "`event_guard` is the v2 candidate: the maximum calibration-percentile rank of invariant, disagreement, and support. "
        "`event_guard_no_invariant` replaces invariant with uncertainty, so it needs no full-state signal. "
        "`margin_vs_primary` and `margin_vs_calibration_simple` use the simple baseline chosen on calibration rows. "
        "Only eligible models enter the averages. A guard is not computed when one of its inputs is missing."
    )
    text = f"""# v3 Stage {stage} report

Label: {label}. {reading}

The margin is the random false-accept rate minus the judge false-accept rate. A positive margin means the judge accepts fewer bad scenarios than that random ranking. This is not the v2 `absolute_margin`, which selects the lowest test-set false-accept rate among simple and stronger baselines.

## Provenance

- git_commit: `{meta.get("git_commit", "unknown")}`
- git_dirty: `{bool(meta.get("git_dirty", False))}`
- protocol_sha256: `{meta.get("protocol_sha256", "")}`
- rmse_metric: {metrics}

## Eligibility

{eligibility_note}

{_markdown_table(eligibility, eligibility_cols)}

## Primary coverages (Q1 and Q2)

Bootstrap of the seed-level mean margin of eligible models, for `{PRIMARY_JUDGE}` only, separately at every primary coverage. `mean_relative_margin` is the ratio of means (`mean_margin / mean_random_far`), matching Q3. One seed cannot make an interval. The width limit at 5% is {float(config["interval_width_limit"]):.2f}. A row is degenerate when under {float(config.get("degenerate_share", DEFAULT_DEGENERATE_SHARE)):.0%} or over {1 - float(config.get("degenerate_share", DEFAULT_DEGENERATE_SHARE)):.0%} of scenarios are bad. Its margin is fixed by the labels, so it does not count toward the check. `thin_models` counts the eligible models whose own bad share is under or over the same limits. Their margins are fixed by the labels and pull the average toward zero. Degenerate 5% rows: {degenerate_rows}. All informative 5% rows narrower than the limit, with one on every system: {precision_ok}.

{_markdown_table(intervals, ["system_id", "badness_target", "coverage", "n_seeds", "bad_share", "degenerate", "n_models", "thin_models", "mean_margin", "mean_relative_margin", "bootstrap_low", "bootstrap_high", "interval_width", "narrower_than_limit"])}

## Settings used

{settings_note}

- horizon: {config["data"]["horizon"]}
- ensemble members: {config["model_runtime"]["ensemble_members"]}
- training settings: {runtime_text}
- `bad_rmse` threshold per system: {thresholds}
- `rmse_metric` per system: {metrics}
- seeds: {", ".join(str(seed) for seed in config["seeds"])}
- minimum judge-test scenarios: {config["min_test_scenarios"]}

## Descriptive coverages

Descriptive only. These coverages are not part of any decision rule, and nothing was selected or tuned on them. They show how the judges compare where fewer in-distribution scenarios are left to accept. Each column averages over eligible models for which that judge is defined (for example, `uncertainty_only` and `event_guard_no_invariant` cover models with native epistemic uncertainty only).

{_descriptive_lines(results)}

## Event guard at primary coverages (Q3)

{guard_intro}

{_guard_lines(results, config=config)}
"""
    path.write_text(text, encoding="utf-8")


def stage_b_blockers(config: dict[str, Any]) -> list[str]:
    """Decisions the freeze still has to make. A Stage B run does not guess them."""
    from scs.systems.bsm1 import confirmatory_window_capacity

    blockers = []
    if not protocol_is_frozen():
        blockers.append("docs/v3/protocol.md is still a draft. The freeze commit has to say the protocol is frozen.")
    if config.get("protocol_frozen") is not True:
        blockers.append("configs/v3/stage_b.yaml has protocol_frozen set to false.")
    if _tracked_working_tree_dirty():
        blockers.append(
            "The git working tree has uncommitted changes to tracked files. Commit the protocol freeze before running Stage B."
        )
    guard = config.get("event_guard_on_bsm1")
    guards = event_guard_definitions(config)
    if not guard:
        blockers.append(
            "The BSM1 refusal guard is unset. Q3 has no definition until the protocol names the guard "
            "that runs when invariant and repair are absent."
        )
    elif guard not in guards:
        blockers.append(f"The BSM1 refusal guard {guard!r} is not one of the guards this config defines: {sorted(guards)}.")
    elif any(column in FULL_STATE_RISK_COLUMNS for column in guards[guard]):
        blockers.append(f"The BSM1 refusal guard {guard!r} uses a signal BSM1 does not have: {guards[guard]}.")
    coverage_rule = config.get("coverage_rule")
    if not coverage_rule:
        blockers.append(
            "How the 5% and 10% coverages combine into one yes is unset. "
            "The protocol has to say both, either, or a pooled margin."
        )
    elif str(coverage_rule) not in ALLOWED_COVERAGE_RULES:
        blockers.append(
            f"coverage_rule {coverage_rule!r} is unsupported; expected one of {sorted(ALLOWED_COVERAGE_RULES)}."
        )
    try:
        rmse_metric_for(config, "bsm1")
    except ValueError as exc:
        blockers.append(str(exc))
    threshold = config.get("rmse_threshold", {})
    bsm1_threshold = threshold.get("bsm1") if isinstance(threshold, dict) else threshold
    if bsm1_threshold is None:
        blockers.append(
            "The BSM1 bad_rmse threshold is unset. The written rule is the smallest whole multiple of the "
            "99th percentile of the best model's per-scenario RMSE on judge_calibration_id, over seeds 0-2, "
            "that no eligible model's in-distribution calibration scenario exceeds."
        )
    capacity = confirmatory_window_capacity()
    n_test_per_split = int(config.get("data", {}).get("n_test_per_split", 0))
    min_scenarios = int(config.get("min_test_scenarios", 2000))
    overlapping = config.get("overlapping_test_windows")
    if overlapping is None:
        blockers.append(
            "The sample-size rule asks for "
            f"{min_scenarios} judge-test scenarios per seed and model. "
            f"The reserved dry block holds {capacity['dry_nonoverlapping']} non-overlapping "
            f"{capacity['span_minutes']}-minute windows, and the rain event holds "
            f"{capacity['rain_nonoverlapping']}. Reaching the sample size means overlapping windows. "
            "The freeze has to say whether those count."
        )
    elif not isinstance(overlapping, bool):
        blockers.append(f"overlapping_test_windows must be a boolean, got {overlapping!r}.")
    elif overlapping is False and n_test_per_split > min(
        capacity["dry_nonoverlapping"], capacity["rain_nonoverlapping"]
    ):
        blockers.append(
            f"overlapping_test_windows is false, but n_test_per_split={n_test_per_split} exceeds the "
            f"non-overlapping capacity ({capacity['dry_nonoverlapping']} dry, {capacity['rain_nonoverlapping']} rain)."
        )
    if 4 * n_test_per_split < min_scenarios:
        blockers.append(
            f"Four BSM1 test splits of {n_test_per_split} give {4 * n_test_per_split} scenarios, "
            f"below min_test_scenarios={min_scenarios}."
        )
    influent_choice = config.get("control_split_influent")
    if not influent_choice:
        blockers.append(
            "The control-only confirmatory split has no influent block. Reserved dry days 5.5-7 are the only "
            "unused dry influent. The calibration control split already uses days 4.5-5.5. Rain days 8-10 "
            "are the rain event."
        )
    elif str(influent_choice) not in ALLOWED_CONTROL_SPLIT_INFLUENTS:
        blockers.append(
            f"control_split_influent {influent_choice!r} is unsupported; "
            f"expected one of {sorted(ALLOWED_CONTROL_SPLIT_INFLUENTS)}."
        )
    return blockers


def _recompute_disagreement_rows(
    rows: list[dict[str, Any]],
    predictions_by_split: dict[str, dict[str, np.ndarray]],
    cohort: list[str],
    state_scale: tuple[np.ndarray, np.ndarray] | None,
) -> None:
    """Update ``disagreement_score`` on scored calibration rows using the calibration-selected cohort."""
    disagreement_by_split: dict[str, np.ndarray] = {}
    for split, predictions in predictions_by_split.items():
        active = [model_id for model_id in cohort if model_id in predictions]
        if len(active) < 2:
            active = list(predictions)
        disagreement_by_split[split] = _batch_disagreement([predictions[m] for m in active], state_scale=state_scale)
    for row in rows:
        split = str(row["split"])
        scenario_index = int(str(row["scenario_id"]).rsplit("_", 1)[-1])
        row["disagreement_score"] = float(disagreement_by_split[split][scenario_index])


def _run_system_two_phase(
    system_id: str,
    config: dict[str, Any],
    event_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit models and validate calibration eligibility + threshold share BEFORE touching confirmatory test splits."""
    stage = str(config.get("stage", "A"))
    n_uncertainty = int(config["data"].get("uncertainty_samples", 5))
    dt = float(config["data"]["dt"])
    use_standardized = rmse_metric_for(config, system_id) == "standardized_state"
    # Offset EnsembleMLPModel member 0 in v3 so it does not duplicate standalone mlp_state_space,
    # without changing v2 defaults.
    runtime = dict(config.get("model_runtime", {}))
    runtime.setdefault("ensemble_seed_offset", 101)
    model_config = {**config, "model_runtime": runtime}
    seed_states: list[dict[str, Any]] = []
    cal_rows_all: list[dict[str, Any]] = []

    for seed in config["seeds"]:
        seed_int = int(seed)
        print(f"stage {stage} {system_id} seed {seed_int}", flush=True)
        if system_id == "bsm1":
            from scs.systems.bsm1 import generate_bsm1_pretest_stage_dataset

            pretest = generate_bsm1_pretest_stage_dataset(config, seed_int)
            held_test: dict[str, Any] | None = None
        else:
            full = generate_v2_dataset(system_id, config, seed=seed_int)
            pretest = {split: batch for split, batch in full.items() if _split_role(split) != "judge_test"}
            held_test = {split: batch for split, batch in full.items() if _split_role(split) == "judge_test"}
        models = {
            model_id: _make_v2_model(model_id, seed_int, model_config, system_id=system_id)
            for model_id in config["models"]
        }
        for model in models.values():
            model.fit(pretest["model_train"])
        support = SupportDistance()
        support.fit(pretest["model_train"])
        state_scale = fit_state_scale(pretest["model_train"].states) if use_standardized else None
        cal_batches = {split: batch for split, batch in pretest.items() if _split_role(split) == "judge_calibration"}
        seed_cal_rows: list[dict[str, Any]] = []
        cal_preds: dict[str, dict[str, np.ndarray]] = {}
        for split, batch in cal_batches.items():
            rows, preds = _rollout_batch(
                system_id,
                seed_int,
                split,
                batch,
                models,
                support,
                event_config,
                dt,
                n_uncertainty,
                state_scale=state_scale,
                keep_predictions=True,
            )
            seed_cal_rows.extend(rows)
            cal_preds[split] = preds
        cal_rows_all.extend(seed_cal_rows)
        seed_states.append(
            {
                "seed": seed_int,
                "models": models,
                "support": support,
                "state_scale": state_scale,
                "cal_batches": cal_batches,
                "cal_rows": seed_cal_rows,
                "cal_preds": cal_preds,
                "held_test": held_test,
            }
        )

    cal_frame = pd.DataFrame(cal_rows_all)
    sys_eligibility = eligibility_table(cal_frame, config)
    if config.get("check_threshold_share"):
        threshold_share_check(cal_frame, sys_eligibility, {**config, "systems": [system_id]})

    cohort = disagreement_cohort(system_id, sys_eligibility, config)
    all_rows: list[dict[str, Any]] = []
    minimum = int(config["min_test_scenarios"])
    for state in seed_states:
        seed_int = int(state["seed"])
        _recompute_disagreement_rows(state["cal_rows"], state["cal_preds"], cohort, state["state_scale"])
        state["cal_preds"].clear()
        all_rows.extend(state["cal_rows"])
        if system_id == "bsm1":
            from scs.systems.bsm1 import generate_bsm1_test_stage_dataset

            test_batches = generate_bsm1_test_stage_dataset(config, seed_int)
        else:
            test_batches = state["held_test"] or {}
        overlap = split_overlap_report({**state["cal_batches"], **test_batches})
        if overlap["overlap_count"] > 0:
            raise RuntimeError(f"calibration/test overlap for {system_id} seed {seed_int}")
        seed_test_rows: list[dict[str, Any]] = []
        for split, batch in test_batches.items():
            if _split_role(split) != "judge_test":
                continue
            rows, _preds = _rollout_batch(
                system_id,
                seed_int,
                split,
                batch,
                state["models"],
                state["support"],
                event_config,
                dt,
                n_uncertainty,
                state_scale=state["state_scale"],
                disagreement_models=cohort,
                keep_predictions=False,
            )
            seed_test_rows.extend(rows)
        test_frame = pd.DataFrame(seed_test_rows)
        if test_frame.empty:
            raise RuntimeError(f"{system_id} seed {seed_int} has fewer than {minimum} judge-test scenarios: {{}}")
        test_counts = test_frame.groupby("model_id").size()
        short = {model_id: int(count) for model_id, count in test_counts.items() if int(count) < minimum}
        if short:
            raise RuntimeError(f"{system_id} seed {seed_int} has fewer than {minimum} judge-test scenarios: {short}")
        all_rows.extend(seed_test_rows)
        state.clear()

    return pd.DataFrame(all_rows), sys_eligibility


def run_v3_stage(config_path: str | Path, output: str | Path) -> dict[str, Any]:
    config = load_v3_config(config_path)
    stage = str(config.get("stage", "A"))
    if stage == "B":
        blockers = stage_b_blockers(config)
        if blockers:
            joined = "\n- ".join(blockers)
            raise RuntimeError(f"Stage B is not run before the freeze:\n- {joined}")
    event_config = load_event_config(config["event_config"])
    metadata = run_metadata()
    frames = []
    eligibilities = []
    for system_id in config["systems"]:
        sys_frame, sys_eligibility = _run_system_two_phase(str(system_id), config, event_config)
        frames.append(sys_frame)
        eligibilities.append(sys_eligibility)
    scored = pd.concat(frames, ignore_index=True)
    eligibility = pd.concat(eligibilities, ignore_index=True)
    results = result_rows_for_frame(scored, config, eligibility)
    results["git_commit"] = metadata["git_commit"]
    results["git_dirty"] = metadata["git_dirty"]
    results["protocol_sha256"] = metadata["protocol_sha256"]
    intervals = interval_table(results, config)
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"stage_{stage.lower().replace('-', '_')}"
    results_path = out_dir / f"{stem}_results.csv"
    report_path = out_dir / f"{stem}_report.md"
    results.to_csv(results_path, index=False)
    _write_report(report_path, eligibility, intervals, results, config, metadata=metadata)
    learned = eligibility[eligibility["model_id"].isin(config["learned_models"])]
    systems_passed = int(learned.loc[learned["eligible"], "system_id"].nunique())
    precision_pass, degenerate_rows = precision_check(intervals)
    return {
        "results": str(results_path),
        "report": str(report_path),
        "eligible_systems": systems_passed,
        "precision_pass": precision_pass,
        "degenerate_rows": degenerate_rows,
        "rows": int(len(results)),
        "git_commit": metadata["git_commit"],
        "git_dirty": metadata["git_dirty"],
        "protocol_sha256": metadata["protocol_sha256"],
    }
