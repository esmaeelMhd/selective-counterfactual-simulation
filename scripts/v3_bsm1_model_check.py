"""Exploratory BSM1 model check on calibration rows.

This does not score judge_test and does not decide eligibility.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from scs.data.schemas import load_dataset
from scs.metrics.trajectory import rmse
from scs.models.hold_last import HoldLastModel
from scs.models.linear_narx import LinearNARXModel
from scs.models.lstm_world import LSTMWorldModel
from scs.models.mlp_state_space import MLPStateSpaceModel
from scs.models.recurrent_world import RecurrentWorldModel
from scs.systems.base import TrajectoryBatch
from scs.systems.bsm1 import REACTOR_WIDTH, effluent_measurements, rollout_with_known_hydraulics

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
    return {
        "state": rmse(predicted, batch.states),
        "one_step": rmse(predicted[:, 1], batch.states[:, 1]),
        "effluent": rmse(measured_pred, measured_true),
        "flow": rmse(predicted[..., _FLOW], batch.states[..., _FLOW]),
        "temperature": rmse(predicted[..., _TEMPERATURE], batch.states[..., _TEMPERATURE]),
        "settler": rmse(predicted[..., _SETTLER], batch.states[..., _SETTLER]),
        "concentration": rmse(predicted[..., _CONCENTRATIONS], batch.states[..., _CONCENTRATIONS]),
    }


def main() -> None:
    data = load_dataset("results/v3/bsm1_pretest")
    train = data["model_train"]
    calibration = data["judge_calibration_id"]
    models = [
        HoldLastModel(),
        LinearNARXModel(),
        LSTMWorldModel(random_state=0, epochs=15),
        RecurrentWorldModel(random_state=0, epochs=15),
        MLPStateSpaceModel(random_state=0, max_iter=80),
    ]
    lines = [
        "# Exploratory BSM1 model check",
        "",
        "Scored on `judge_calibration_id` after fitting `model_train`.",
        "This is not the eligibility rule. Eligibility uses `judge_test_id`, and that split is not generated.",
        "Reactor and effluent flow are set from the influent with the open-loop recycle and waste flows.",
        "",
        "| model | fit seconds | state RMSE | one-step RMSE | flow | temperature | settler | concentration | effluent RMSE | final train loss |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in models:
        started = time.perf_counter()
        model.fit(train)
        fit_seconds = time.perf_counter() - started
        scored = _score(model, calibration)
        final_loss = getattr(model, "train_losses", [float("nan")])[-1]
        lines.append(
            "| {model_id} | {fit_seconds:.1f} | {state:.4g} | {one_step:.4g} | {flow:.4g} | {temperature:.4g} | {settler:.4g} | {concentration:.4g} | {effluent:.4g} | {final_loss:.4g} |".format(
                model_id=model.model_id,
                fit_seconds=fit_seconds,
                final_loss=final_loss,
                **scored,
            )
        )
        print(lines[-1], flush=True)
    output = Path("results/v3/bsm1_model_check/report.md")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
