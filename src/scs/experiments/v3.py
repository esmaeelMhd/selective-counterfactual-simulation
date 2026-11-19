"""Stage A runner for the v3 refusal benchmark.

The run is exploratory. It does not answer Q1–Q3. Margins are against a
seeded random ranking. They are not the v2 margin, which picks the lowest
test-set false-accept rate among simple and stronger baselines.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from scs.experiments.registry import make_system
from scs.experiments.v2 import (
    SIGNAL_COLUMNS,
    SIMPLE_RISK_COLUMN,
    _fit_v2_calibrated_scores,
    _low_coverage_far,
    _make_v2_model,
    _risk_curve_from_labels,
    _split_role,
    generate_v2_dataset,
    load_event_config,
    split_overlap_report,
    trajectory_bad_event,
)
from scs.experiments.v2_event_risk import event_guarded_score
from scs.signals import available_signals
from scs.validators.invariants import invariant_residual_score
from scs.validators.repair import repair_amount_score
from scs.validators.support import SupportDistance

PRIMARY_JUDGE = "calibration_selected_candidate_ranker"
RANDOM_JUDGE = "random_baseline"


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
        "event_guard_inputs",
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
    threshold = data["rmse_threshold"]
    if isinstance(threshold, dict):
        absent = [str(name) for name in data["systems"] if name not in threshold]
        if absent:
            raise ValueError(f"rmse_threshold has no value for systems: {absent}")
    members = int(data["model_runtime"].get("ensemble_members", 0))
    if members < 5:
        raise ValueError("Stage A requires an ensemble of at least 5 members")
    return data


DEFAULT_DEGENERATE_SHARE = 0.01


def rmse_threshold_for(config: dict[str, Any], system_id: str) -> float:
    """The ``bad_rmse`` threshold for one system. A single number applies to every system."""
    value = config["rmse_threshold"]
    return float(value[system_id]) if isinstance(value, dict) else float(value)


def _batch_disagreement(predictions: list[np.ndarray]) -> np.ndarray:
    if len(predictions) < 2:
        return np.zeros(predictions[0].shape[0], dtype=float)
    distances = []
    for left_index, left in enumerate(predictions):
        for right in predictions[left_index + 1 :]:
            distances.append(np.sqrt(np.mean((left - right) ** 2, axis=(1, 2))))
    return np.mean(np.stack(distances, axis=0), axis=0)


def _support_scores(support: SupportDistance, actions: np.ndarray, disturbances: np.ndarray) -> np.ndarray:
    if support.mean_ is None or support.std_ is None:
        raise RuntimeError("support distance must be fit before scoring")
    features = np.concatenate([actions, disturbances], axis=-1)
    scaled = (features - support.mean_) / support.std_
    return np.mean(np.linalg.norm(scaled, axis=-1), axis=1)


def _uncertainty_scores(
    model: Any,
    initial_states: np.ndarray,
    actions: np.ndarray,
    disturbances: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
    history: np.ndarray | None = None,
) -> np.ndarray:
    """Spread of sampled rollouts, or NaN when the model has no way to sample.

    Ensembles use their members. Other models add noise with their fitted
    residual spread. A model with neither returns NaN, which the judges treat
    as a missing signal. A model that predicts no change has no spread and
    returns zeros.
    """
    n_rows, horizon, _action_dim = actions.shape
    if model.model_id == "hold_last" or n_samples <= 0:
        return np.zeros(n_rows, dtype=float)
    samples = []
    members = getattr(model, "members", None)
    if members:
        for sample_index in range(n_samples):
            member = members[sample_index % len(members)]
            samples.append(member.predict_rollouts(initial_states, actions, disturbances, history=history))
    else:
        residual_std = getattr(model, "residual_std", None)
        if residual_std is None:
            return np.full(n_rows, np.nan, dtype=float)
        for _sample_index in range(n_samples):
            noise = rng.normal(0.0, residual_std, size=(n_rows, horizon, initial_states.shape[1]))
            samples.append(model.predict_rollouts(initial_states, actions, disturbances, history=history, noise=noise))
    stacked = np.stack(samples, axis=0)
    return np.mean(np.std(stacked, axis=0), axis=(1, 2))


def _score_batch(
    system_id: str,
    seed: int,
    split: str,
    batch: Any,
    models: dict[str, Any],
    support: SupportDistance,
    event_config: dict[str, Any],
    dt: float,
    n_uncertainty: int,
) -> list[dict[str, Any]]:
    system = make_system(system_id)
    initial = batch.states[:, 0, :]
    predictions = {
        model_id: model.predict_rollouts(initial, batch.actions, batch.disturbances, history=batch.measured_history)
        for model_id, model in models.items()
    }
    disagreement = _batch_disagreement([predictions[model_id] for model_id in models])
    support_scores = _support_scores(support, batch.actions, batch.disturbances)
    uncertainty = {
        model_id: _uncertainty_scores(
            model,
            initial,
            batch.actions,
            batch.disturbances,
            n_uncertainty,
            np.random.default_rng(10_000 + seed * 100 + sum(ord(char) for char in model_id)),
            history=batch.measured_history,
        )
        for model_id, model in models.items()
    }
    true_events = [
        trajectory_bad_event(system_id, batch.states[index], event_config) for index in range(batch.n_trajectories)
    ]
    rows: list[dict[str, Any]] = []
    role = _split_role(split)
    for index in range(batch.n_trajectories):
        true_states = batch.states[index]
        for model_id, predicted in predictions.items():
            pred = predicted[index]
            pred_event = trajectory_bad_event(system_id, pred, event_config)
            diff = pred - true_states
            signals_here = available_signals(system_id)
            rows.append(
                {
                    "system_id": system_id,
                    "seed": seed,
                    "split": split,
                    "role": role,
                    "scenario_type": batch.scenario_type[index],
                    "scenario_id": f"{system_id}_seed{seed}_{split}_{index:04d}",
                    "model_id": model_id,
                    "rmse": float(np.sqrt(np.mean(diff**2))),
                    "true_event": bool(true_events[index]),
                    "pred_event": bool(pred_event),
                    "event_mismatch": bool(true_events[index] != pred_event),
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
    return rows


def _prediction_frame(system_id: str, seed: int, config: dict[str, Any], event_config: dict[str, Any]) -> pd.DataFrame:
    dataset = generate_v2_dataset(system_id, config, seed=seed)
    overlap = split_overlap_report(dataset)
    if overlap["overlap_count"] > 0:
        raise RuntimeError(f"calibration/test overlap for {system_id} seed {seed}")
    models = {model_id: _make_v2_model(model_id, seed, config) for model_id in config["models"]}
    for model in models.values():
        model.fit(dataset["model_train"])
    support = SupportDistance()
    support.fit(dataset["model_train"])
    rows: list[dict[str, Any]] = []
    n_uncertainty = int(config["data"].get("uncertainty_samples", 2))
    dt = float(config["data"]["dt"])
    for split, batch in dataset.items():
        if _split_role(split) not in {"judge_calibration", "judge_test"}:
            continue
        rows.extend(_score_batch(system_id, seed, split, batch, models, support, event_config, dt, n_uncertainty))
    frame = pd.DataFrame(rows)
    test_counts = frame.loc[frame["role"] == "judge_test"].groupby("model_id").size()
    minimum = int(config["min_test_scenarios"])
    short = {model_id: int(count) for model_id, count in test_counts.items() if int(count) < minimum}
    if short:
        raise RuntimeError(f"{system_id} seed {seed} has fewer than {minimum} judge-test scenarios: {short}")
    return frame


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
            beats = learned and all(float(row.id_rmse) < baseline for baseline in baselines.values())
            rows.append(
                {
                    "system_id": system_id,
                    "model_id": row.model_id,
                    "id_rmse": float(row.id_rmse),
                    "eligible": bool(beats),
                    **{f"baseline_{name}_id_rmse": value for name, value in baselines.items()},
                }
            )
    return pd.DataFrame(rows)


def _relative_margin(absolute: float, baseline_far: float) -> float:
    if baseline_far > 0.0 and math.isfinite(baseline_far):
        return float(absolute / baseline_far)
    return float("nan")


def _finite_signal_names(frame: pd.DataFrame) -> list[str]:
    names = []
    for name, column in SIMPLE_RISK_COLUMN.items():
        values = frame[column].to_numpy(dtype=float)
        if np.isfinite(values).all():
            names.append(name)
    return names


def _rank_mean_available(calibration: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Calibration-percentile mean over the signals that are actually present."""
    labels = calibration["bad_label"].to_numpy(dtype=bool)
    parts = []
    for column in SIGNAL_COLUMNS:
        cal_values = calibration[column].to_numpy(dtype=float)
        test_values = test[column].to_numpy(dtype=float)
        if not np.isfinite(cal_values).all() or not np.isfinite(test_values).all():
            continue
        orientation = 1
        if 0 < int(labels.sum()) < len(labels):
            orientation = 1 if float(cal_values[labels].mean()) >= float(cal_values[~labels].mean()) else -1
        reference = np.sort(orientation * cal_values)
        score = orientation * test_values
        parts.append(np.searchsorted(reference, score, side="right") / max(len(reference), 1))
    if not parts:
        raise RuntimeError("no finite refusal signal")
    return np.mean(np.vstack(parts), axis=0)


def _best_simple_baseline(calibration: pd.DataFrame, coverages: list[float]) -> str:
    labels = calibration["bad_label"].to_numpy(dtype=bool)
    best_name = ""
    best_key: tuple[float, str] | None = None
    finite_names = _finite_signal_names(calibration)
    if not finite_names:
        raise RuntimeError("no finite simple baseline")
    for name in finite_names:
        column = SIMPLE_RISK_COLUMN[name]
        far = _low_coverage_far(labels, calibration[column].to_numpy(dtype=float), coverages)
        key = (far, name)
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
    eligible = {
        (row.system_id, row.model_id): bool(row.eligible) for row in eligibility.itertuples(index=False)
    }
    pieces = []
    for (system_id, seed), group in frame.groupby(["system_id", "seed"], sort=True):
        threshold = rmse_threshold_for(config, str(system_id))
        for target in config["badness_targets"]:
            labelled = group.copy()
            if target == "bad_rmse":
                labelled["bad_label"] = labelled["rmse"].astype(float) > threshold
            elif target == "bad_event":
                labelled["bad_label"] = labelled["event_mismatch"].astype(bool)
            else:
                raise ValueError(f"unsupported badness target: {target}")
            for column, signal in SIMPLE_RISK_COLUMN.items():
                labelled[f"risk_{column}"] = labelled[signal]
            calibration = labelled[labelled["role"] == "judge_calibration"].copy()
            test = labelled[labelled["role"] == "judge_test"].copy()
            test["risk_random_baseline"] = _assign_random(test, int(seed))
            if all(np.isfinite(labelled[column].to_numpy(dtype=float)).all() for column in SIGNAL_COLUMNS):
                calibrated = _fit_v2_calibrated_scores(calibration, test, coverages)
                primary_scores = calibrated[PRIMARY_JUDGE]
            else:
                primary_scores = _rank_mean_available(calibration, test)
            test["risk_calibration_selected_candidate_ranker"] = primary_scores
            for model_id, model_test in test.groupby("model_id", sort=True):
                model_calibration = calibration[calibration["model_id"] == model_id]
                simple_name = _best_simple_baseline(model_calibration, coverages)
                guard_ready = all(
                    np.isfinite(model_calibration[column].to_numpy(dtype=float)).all()
                    and np.isfinite(model_test[column].to_numpy(dtype=float)).all()
                    for column in config["event_guard_inputs"]
                )
                judge_scores = {
                    name: model_test[f"risk_{name}"].to_numpy(dtype=float)
                    for name in _finite_signal_names(model_test)
                }
                judge_scores[RANDOM_JUDGE] = model_test["risk_random_baseline"].to_numpy(dtype=float)
                judge_scores[PRIMARY_JUDGE] = model_test["risk_calibration_selected_candidate_ranker"].to_numpy(dtype=float)
                # The guard is defined by all of its inputs. When one is missing it is not
                # computed on a smaller set: the system has no guard until the protocol names one.
                if guard_ready:
                    judge_scores["event_guard"] = event_guarded_score(
                        model_calibration, model_test, list(config["event_guard_inputs"])
                    )
                labels = model_test["bad_label"].to_numpy(dtype=bool)
                random_curve = _risk_curve_from_labels(labels, judge_scores[RANDOM_JUDGE], coverages)
                primary_curve = _risk_curve_from_labels(labels, judge_scores[PRIMARY_JUDGE], coverages)
                simple_curve = _risk_curve_from_labels(labels, judge_scores[simple_name], coverages)
                primary_far = {float(row.coverage): float(row.false_accept_rate) for row in primary_curve.itertuples(index=False)}
                simple_far = {float(row.coverage): float(row.false_accept_rate) for row in simple_curve.itertuples(index=False)}
                for judge_id, scores in judge_scores.items():
                    curve = _risk_curve_from_labels(labels, scores, coverages)
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
    """Bootstrap the seed-level mean margin of eligible models, separately at each coverage.

    A row is degenerate when almost none or almost all test scenarios are bad. The
    margin is then fixed near zero by the labels, so a narrow interval says nothing.
    """
    primary = results[(results["judge_id"] == PRIMARY_JUDGE) & (results["eligible"])].copy()
    rng = np.random.default_rng(int(config["bootstrap_seed"]))
    rows = []
    if primary.empty:
        return pd.DataFrame(rows)
    grouped = primary.groupby(["system_id", "badness_target", "coverage", "seed"], as_index=False).agg(
        absolute_margin=("absolute_margin", "mean"), bad_share=("bad_share", "mean")
    )
    limit = float(config["interval_width_limit"])
    n_samples = int(config["bootstrap_samples"])
    edge = float(config.get("degenerate_share", DEFAULT_DEGENERATE_SHARE))
    for (system_id, target, coverage), group in grouped.groupby(["system_id", "badness_target", "coverage"], sort=True):
        low, high = bootstrap_interval(group["absolute_margin"].to_numpy(dtype=float), rng, n_samples)
        width = high - low if math.isfinite(low) and math.isfinite(high) else float("nan")
        share = float(group["bad_share"].mean())
        rows.append(
            {
                "system_id": system_id,
                "badness_target": target,
                "coverage": float(coverage),
                "n_seeds": int(group["seed"].nunique()),
                "bad_share": share,
                "degenerate": bool(share < edge or share > 1.0 - edge),
                "mean_margin": float(group["absolute_margin"].mean()),
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


def _guard_lines(results: pd.DataFrame) -> str:
    """Event guard against the primary ranker and the calibration-chosen simple baseline.

    Only eligible models enter, so a model that is worse than the baselines does
    not move the averages. Systems with no guard rows or no eligible model are named.
    """
    systems = sorted(results["system_id"].unique())
    at_five = results[np.isclose(results["coverage"], 0.05)]
    guard = at_five[(at_five["judge_id"] == "event_guard") & at_five["eligible"]].copy()
    notes = []
    no_guard = [name for name in systems if name not in set(at_five.loc[at_five["judge_id"] == "event_guard", "system_id"])]
    if no_guard:
        notes.append(f"No event guard is defined on: {', '.join(no_guard)}. A required guard input is missing there.")
    no_eligible = [name for name in systems if name not in set(guard["system_id"]) and name not in no_guard]
    if no_eligible:
        notes.append(f"No eligible model, so no guard row: {', '.join(no_eligible)}.")
    if guard.empty:
        return "\n\n".join(["_none_", *notes])
    guard["margin_vs_primary"] = guard["primary_far"] - guard["false_accept_rate"]
    guard["margin_vs_calibration_simple"] = guard["calibration_simple_far"] - guard["false_accept_rate"]
    summary = (
        guard.groupby(["system_id", "badness_target"], as_index=False)
        .agg(
            n_models=("model_id", "nunique"),
            event_guard_far=("false_accept_rate", "mean"),
            primary_far=("primary_far", "mean"),
            calibration_simple_far=("calibration_simple_far", "mean"),
            margin_vs_primary=("margin_vs_primary", "mean"),
            margin_vs_calibration_simple=("margin_vs_calibration_simple", "mean"),
        )
    )
    table = _markdown_table(
        summary,
        [
            "system_id",
            "badness_target",
            "n_models",
            "event_guard_far",
            "primary_far",
            "calibration_simple_far",
            "margin_vs_primary",
            "margin_vs_calibration_simple",
        ],
    )
    return "\n\n".join([table, *notes])


def _write_report(
    path: Path,
    eligibility: pd.DataFrame,
    intervals: pd.DataFrame,
    results: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    learned = eligibility[eligibility["model_id"].isin(config["learned_models"])]
    systems_passed = sorted(learned.loc[learned["eligible"], "system_id"].unique())
    at_five = intervals[np.isclose(intervals["coverage"], 0.05)] if not intervals.empty else intervals
    precision_ok, degenerate_rows = precision_check(intervals)
    runtime_text = ", ".join(
        f"{key} {value}" for key, value in sorted(config["model_runtime"].items()) if key != "ensemble_members"
    )
    thresholds = ", ".join(f"{name} {rmse_threshold_for(config, str(name)):g}" for name in config["systems"])
    text = f"""# v3 Stage A report

Label: exploratory. This report does not answer Q1, Q2, or Q3.

The margin is the random false-accept rate minus the judge false-accept rate. A positive margin means the judge accepts fewer bad scenarios than that random ranking. This is not the v2 `absolute_margin`, which selects the lowest test-set false-accept rate among simple and stronger baselines.

## Eligibility

Working rule for this run: mean RMSE on `{eligibility_split(config)}` is strictly below both `hold_last` and `linear_narx` on that system. Eligibility does not read judge-test rows. Learned models passing: {len(systems_passed)} system(s) ({", ".join(systems_passed) if len(systems_passed) else "none"}). The plan's Stage A check wants at least two systems.

{_markdown_table(eligibility, ["system_id", "model_id", "id_rmse", "baseline_hold_last_id_rmse", "baseline_linear_narx_id_rmse", "eligible"])}

## Interval at 5% coverage

Bootstrap of the seed-level mean margin of eligible models, for `{PRIMARY_JUDGE}` only. One seed cannot make an interval. The width limit is {float(config["interval_width_limit"]):.2f}. A row is degenerate when under {float(config.get("degenerate_share", DEFAULT_DEGENERATE_SHARE)):.0%} or over {1 - float(config.get("degenerate_share", DEFAULT_DEGENERATE_SHARE)):.0%} of scenarios are bad. Its margin is fixed by the labels, so it does not count toward the check. Degenerate 5% rows: {degenerate_rows}. All informative 5% rows narrower than the limit, with one on every system: {precision_ok}.

{_markdown_table(at_five, ["system_id", "badness_target", "n_seeds", "bad_share", "degenerate", "mean_margin", "bootstrap_low", "bootstrap_high", "interval_width", "narrower_than_limit"])}

## Settings used

These numbers are Stage A choices. They are not the freeze.

- horizon: {config["data"]["horizon"]}
- ensemble members: {config["model_runtime"]["ensemble_members"]}
- training settings: {runtime_text}
- `bad_rmse` threshold per system: {thresholds}
- seeds: {", ".join(str(seed) for seed in config["seeds"])}
- minimum judge-test scenarios: {config["min_test_scenarios"]}

## Event guard at 5% coverage

Exploratory only. The guard is the v2 candidate: the maximum calibration-percentile rank of invariant, disagreement, and support. `margin_vs_primary` and `margin_vs_calibration_simple` use the simple baseline chosen on calibration rows. They are not a Q3 answer. Only eligible models enter the averages. The guard is not computed when one of its inputs is missing. On BSM1 that is the invariant signal, and no replacement guard is written yet.

{_guard_lines(results)}
"""
    path.write_text(text, encoding="utf-8")


def run_v3_stage(config_path: str | Path, output: str | Path) -> dict[str, Any]:
    config = load_v3_config(config_path)
    event_config = load_event_config(config["event_config"])
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for system_id in config["systems"]:
        for seed in config["seeds"]:
            print(f"stage A {system_id} seed {seed}", flush=True)
            frames.append(_prediction_frame(str(system_id), int(seed), config, event_config))
    scored = pd.concat(frames, ignore_index=True)
    eligibility = eligibility_table(scored, config)
    results = result_rows_for_frame(scored, config, eligibility)
    intervals = interval_table(results, config)
    results_path = out_dir / "stage_a_results.csv"
    report_path = out_dir / "stage_a_report.md"
    results.to_csv(results_path, index=False)
    _write_report(report_path, eligibility, intervals, results, config)
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
    }
