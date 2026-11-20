"""Exploratory BSM1 model check on calibration rows.

It applies the working eligibility rule: mean per-scenario state RMSE on the eligibility
split, strictly below both hold_last and linear_narx, with the mean taken over seeds as
Stage A does. The control-shift calibration split is scored as a check and does not decide
eligibility. No judge-test row exists.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from scs.data.schemas import load_dataset
from scs.metrics.trajectory import rmse
from scs.models.ensemble_lstm import EnsembleLSTMModel
from scs.models.hold_last import HoldLastModel
from scs.models.linear_narx import LinearNARXModel
from scs.models.lstm_world import LSTMWorldModel
from scs.models.mlp_state_space import MLPStateSpaceModel
from scs.models.recurrent_world import RecurrentWorldModel
from scs.systems.base import TrajectoryBatch
from scs.systems.bsm1 import (
    REACTOR_WIDTH,
    effluent_measurements,
    hydraulic_projection,
    pretest_overlap_gap,
    rollout_with_known_hydraulics,
)

_FLOW = [14 + REACTOR_WIDTH * block for block in range(6)]
_TEMPERATURE = [15 + REACTOR_WIDTH * block for block in range(6)]
_SETTLER = list(range(6 * REACTOR_WIDTH, 6 * REACTOR_WIDTH + 10))
_CONCENTRATIONS = [index for index in range(6 * REACTOR_WIDTH) if index not in set(_FLOW + _TEMPERATURE)]


def _score(model, batch: TrajectoryBatch) -> dict[str, float]:
    if hasattr(model, "predict_rollouts"):
        predicted = rollout_with_known_hydraulics(model, batch.states[:, 0, :], batch.actions, batch.disturbances)
    else:
        predicted = np.stack(
            [
                model.predict_rollout(batch.states[index, 0], batch.actions[index], batch.disturbances[index])
                for index in range(batch.n_trajectories)
            ]
        )
    measured_pred = np.stack([effluent_measurements(predicted[index]) for index in range(batch.n_trajectories)])
    measured_true = np.stack([effluent_measurements(batch.states[index]) for index in range(batch.n_trajectories)])
    per_scenario = np.sqrt(np.mean((predicted - batch.states) ** 2, axis=(1, 2)))
    return {
        "scenario_mean": float(per_scenario.mean()),
        "state": rmse(predicted, batch.states),
        "one_step": rmse(predicted[:, 1], batch.states[:, 1]),
        "effluent": rmse(measured_pred, measured_true),
        "flow": rmse(predicted[..., _FLOW], batch.states[..., _FLOW]),
        "temperature": rmse(predicted[..., _TEMPERATURE], batch.states[..., _TEMPERATURE]),
        "settler": rmse(predicted[..., _SETTLER], batch.states[..., _SETTLER]),
        "concentration": rmse(predicted[..., _CONCENTRATIONS], batch.states[..., _CONCENTRATIONS]),
    }


SEEDS = (0, 1, 2)
# Training budget rule for the recurrent adapters, fixed before looking at any calibration row.
# The last 15% of the model_train windows are held out, less a gap of 18 windows that keeps them
# apart in time from the rest. Every 5 epochs the free-running rollout error on them is scored,
# with the BSM1 flow assignment, the best weights are kept, and training stops after 3 checks
# without improvement, or at 120 epochs. Neither calibration split is read. The delta MLP keeps
# the 80 iterations it had before this work.
STOPPING_RULE = dict(
    early_stopping=True,
    validation_fraction=0.15,
    validation_gap=pretest_overlap_gap(),
    check_every=5,
    patience=3,
    projection_factory=hydraulic_projection,
)
MODELS = [
    ("hold_last", lambda seed: HoldLastModel(), (0,)),
    ("linear_narx", lambda seed: LinearNARXModel(), (0,)),
    ("mlp_state_space", lambda seed: MLPStateSpaceModel(random_state=seed, max_iter=80), SEEDS),
    ("lstm_world", lambda seed: LSTMWorldModel(random_state=seed, epochs=120, **STOPPING_RULE), SEEDS),
    ("ensemble_lstm", lambda seed: EnsembleLSTMModel(random_state=seed, n_members=5, epochs=120, **STOPPING_RULE), SEEDS),
    ("recurrent_world", lambda seed: RecurrentWorldModel(random_state=seed, epochs=120, **STOPPING_RULE), (0,)),
]


def _selected_epochs(model) -> float:
    members = getattr(model, "members", None)
    if members:
        return float(np.mean([member.selected_epochs for member in members]))
    chosen = getattr(model, "selected_epochs", None)
    return float("nan") if chosen is None else float(chosen)


def main() -> None:
    data = load_dataset("results/v3/bsm1_pretest")
    train = data["model_train"]
    calibration = data["judge_calibration_id"]
    control = data["judge_calibration_control"]
    rows = []
    for name, factory, seeds in MODELS:
        scored, shifted, seconds, losses, epochs = [], [], [], [], []
        for seed in seeds:
            model = factory(seed)
            started = time.perf_counter()
            model.fit(train)
            seconds.append(time.perf_counter() - started)
            scored.append(_score(model, calibration))
            shifted.append(_score(model, control)["scenario_mean"])
            losses.append(getattr(model, "train_losses", [float("nan")])[-1])
            epochs.append(_selected_epochs(model))
            print(name, seed, scored[-1]["scenario_mean"], shifted[-1], flush=True)
        rows.append((name, seeds, scored, shifted, float(np.mean(seconds)), float(np.mean(epochs))))
    by_id = {name: float(np.mean([item["scenario_mean"] for item in scored])) for name, _s, scored, _c, _t, _e in rows}
    by_control = {name: float(np.mean(shifted)) for name, _s, _sc, shifted, _t, _e in rows}
    bar = min(by_id["hold_last"], by_id["linear_narx"])
    control_bar = min(by_control["hold_last"], by_control["linear_narx"])
    lines = [
        "# Exploratory BSM1 model check",
        "",
        "Fit on `model_train`, scored on `judge_calibration_id` (the eligibility split) and on `judge_calibration_control`. No judge-test row exists.",
        "Eligible means the mean per-scenario state RMSE, averaged over seeds, is strictly below both `hold_last` and `linear_narx`, the working rule in `docs/v3/protocol.md`. The control column applies the same bar on the control-shift split and does not decide eligibility.",
        "Each influent window has its own flow and load factors. Splits read disjoint blocks of the 7 unique dry days, and each window starts from its own minute of one continuous dry run.",
        "Reactor and effluent flow are set from the influent with the open-loop recycle and waste flows, inside each model's own rollout, so recurrent models keep their state.",
        "The LSTM and recurrent adapters predict the change in state over one step. The calibration split holds 2 independent daily influent shapes, so these numbers are noisy.",
        "The LSTM, its ensemble and the recurrent model train under a stopping rule fixed before any calibration row was read: the last 15% of the `model_train` windows are held out, less a gap of 18 windows, the free-running rollout error on them is checked every 5 epochs, the best weights are kept, and training stops after 3 checks without improvement or at 120 epochs. Neither calibration split decides a budget. The delta MLP keeps its earlier 80 iterations.",
        "",
        "| model | seeds | fit seconds | mean scenario RMSE | seeds below bar | eligible | control-shift RMSE | control below bar | pooled state RMSE | one-step RMSE | settler | concentration | effluent RMSE | epochs used |",
        "|---|---:|---:|---:|---:|:---:|---:|:---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, seeds, scored, shifted, seconds, chosen in rows:
        reference = name in ("hold_last", "linear_narx")
        below = sum(1 for item in scored if item["scenario_mean"] < bar)
        mean = lambda key: float(np.mean([item[key] for item in scored]))
        lines.append(
            "| {name} | {n} | {seconds:.1f} | {scenario:.4g} | {below} | {eligible} | {control:.4g} | {control_ok} | {state:.4g} | {one_step:.4g} | {settler:.4g} | {concentration:.4g} | {effluent:.4g} | {chosen:.4g} |".format(
                name=name,
                n=len(seeds),
                seconds=seconds,
                scenario=by_id[name],
                below="n/a" if reference else f"{below}/{len(seeds)}",
                eligible="n/a" if reference else str(by_id[name] < bar),
                control=by_control[name],
                control_ok="n/a" if reference else str(by_control[name] < control_bar),
                state=mean("state"),
                one_step=mean("one_step"),
                settler=mean("settler"),
                concentration=mean("concentration"),
                effluent=mean("effluent"),
                chosen=chosen,
            )
        )
    output = Path("results/v3/bsm1_model_check/report.md")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
