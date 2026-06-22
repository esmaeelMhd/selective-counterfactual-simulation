from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

from scs.data.generate import generate_dataset
from scs.metrics.trajectory import mae, rmse
from scs.models.base import flatten_supervised
from scs.models.user_model import UserSimulatorModel
from scs.systems.base import TrajectoryBatch


class DampedLinearUserModel(UserSimulatorModel):
    """Small custom model example for local comparison.

    This is intentionally simple. It is provided to demonstrate the adapter
    contract and is not part of the current benchmark evidence claim.
    """

    model_id = "damped_linear_user"

    def __init__(self, alpha: float = 1e-3, damping: float = 0.92) -> None:
        self.alpha = alpha
        self.damping = damping
        self.regressor = Ridge(alpha=alpha)
        self.state_dim: int | None = None

    def fit(self, train_batch: TrajectoryBatch) -> None:
        features, targets = flatten_supervised(train_batch)
        self.regressor.fit(features, targets)
        self.state_dim = int(train_batch.states.shape[-1])

    def predict_rollout(
        self,
        initial_state: np.ndarray,
        actions: np.ndarray,
        disturbances: np.ndarray,
    ) -> np.ndarray:
        initial_state, actions, disturbances = self.validate_inputs(initial_state, actions, disturbances)
        states = np.empty((len(actions) + 1, len(initial_state)), dtype=float)
        states[0] = initial_state
        for t in range(len(actions)):
            feature = np.concatenate([states[t], actions[t], disturbances[t]], axis=0)[None, :]
            delta = np.asarray(self.regressor.predict(feature)[0], dtype=float)
            states[t + 1] = states[t] + self.damping * delta
        return self.validate_rollout_output(states, initial_state, actions)


def run_example(output: str | Path) -> dict:
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = generate_dataset(
        system_id="two_tank",
        n_train=32,
        n_id_test=8,
        n_ood_test=8,
        horizon=18,
        dt=0.1,
        seed=17,
    )
    model = DampedLinearUserModel()
    model.fit(dataset["train"])
    rows = []
    for split in ["id_test", "ood_action_magnitude", "ood_inflow_spike", "ood_combined"]:
        batch = dataset[split]
        for idx in range(batch.n_trajectories):
            pred = model.predict_rollout(batch.states[idx, 0], batch.actions[idx], batch.disturbances[idx])
            truth = batch.states[idx]
            rows.append(
                {
                    "split": split,
                    "scenario_index": idx,
                    "rmse": rmse(pred, truth),
                    "mae": mae(pred, truth),
                }
            )
    rmse_values = [row["rmse"] for row in rows]
    summary = {
        "model_id": model.model_id,
        "is_evidence_for_current_claim": False,
        "n_scenarios": len(rows),
        "rmse_mean": float(np.mean(rmse_values)),
        "output": str(out_dir),
    }
    (out_dir / "custom_model_example_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "custom_model_smoke.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    report = f"""# Custom Model Example Report

## Model

{model.model_id}

## Result

Mean RMSE: {summary["rmse_mean"]:.6f}

## Claim boundary

This custom model result is local adapter output only and is not evidence for the current supported claim.
"""
    (out_dir / "custom_model_example_report.md").write_text(report, encoding="utf-8")
    (out_dir / "custom_model_report.md").write_text(report, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the custom model adapter example.")
    parser.add_argument("--output", default="results/custom_model_example")
    args = parser.parse_args()
    summary = run_example(args.output)
    print(summary["model_id"])


if __name__ == "__main__":
    main()
