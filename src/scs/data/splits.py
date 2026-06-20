from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scs.systems.base import TrajectoryBatch


@dataclass(frozen=True)
class SplitDifference:
    train_action_range: float
    test_action_range: float
    train_inflow_max: float
    test_inflow_max: float

    @property
    def action_range_increased(self) -> bool:
        return self.test_action_range > self.train_action_range * 1.25

    @property
    def inflow_spike_increased(self) -> bool:
        return self.test_inflow_max > self.train_inflow_max * 1.25


def action_range(batch: TrajectoryBatch) -> float:
    return float(np.max(batch.actions[..., 0]) - np.min(batch.actions[..., 0]))


def max_inflow(batch: TrajectoryBatch) -> float:
    return float(np.max(batch.disturbances[..., 0]))


def compare_split_shift(train: TrajectoryBatch, test: TrajectoryBatch) -> SplitDifference:
    return SplitDifference(
        train_action_range=action_range(train),
        test_action_range=action_range(test),
        train_inflow_max=max_inflow(train),
        test_inflow_max=max_inflow(test),
    )


def assert_not_identical(left: TrajectoryBatch, right: TrajectoryBatch) -> None:
    if left.states.shape == right.states.shape and np.allclose(left.states, right.states):
        raise AssertionError(f"{left.split} and {right.split} states are identical")
    if left.actions.shape == right.actions.shape and np.allclose(left.actions, right.actions):
        raise AssertionError(f"{left.split} and {right.split} actions are identical")

