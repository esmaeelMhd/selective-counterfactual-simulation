from __future__ import annotations

import numpy as np
import pandas as pd

from scs.data.generate import generate_dataset
from scs.experiments.registry import make_model
from scs.validators.invariants import invariant_residual_score
from scs.validators.judges import JUDGE_IDS, compute_judge_score_frame, score_judge
from scs.validators.repair import repair_amount_score
from scs.validators.support import SupportDistance
from scs.systems.two_tank import TwoTankSystem


def test_support_distance_higher_on_ood_action() -> None:
    dataset = generate_dataset("two_tank", 60, 20, 20, horizon=20, dt=0.1, seed=11)
    support = SupportDistance()
    support.fit(dataset["train"])
    id_score = float(np.mean(support.score_batch(dataset["id_test"])))
    ood_score = float(np.mean(support.score_batch(dataset["ood_action_magnitude"])))
    assert ood_score > id_score


def test_invariant_and_repair_scores_detect_bad_trajectories() -> None:
    system = TwoTankSystem()
    actions = np.full((12, 1), 0.7)
    disturbances = np.full((12, 2), [0.5, 0.45])
    states = system.rollout(np.array([4.0, 3.0]), actions, disturbances, dt=0.1)
    clean = invariant_residual_score(system, states, actions, disturbances, dt=0.1)
    corrupted = states.copy()
    corrupted[6:, 0] += 1.5
    dirty = invariant_residual_score(system, corrupted, actions, disturbances, dt=0.1)
    assert clean < 1e-10
    assert dirty > clean + 0.1

    out_of_bounds = states.copy()
    out_of_bounds[3, 0] = -2.0
    out_of_bounds[5, 1] = 12.0
    assert repair_amount_score(system, out_of_bounds) > 0.0


def test_every_judge_returns_finite_score() -> None:
    signals = {
        "support_distance": 1.0,
        "uncertainty": 0.2,
        "disagreement": 0.3,
        "invariant_residual": 0.4,
        "repair_amount": 0.1,
        "error": 0.5,
        "random_baseline": 0.7,
    }
    for judge_id in JUDGE_IDS:
        assert np.isfinite(score_judge(judge_id, signals))

    df = pd.DataFrame([signals, {**signals, "support_distance": 2.0, "error": 0.1}])
    scores = compute_judge_score_frame(df, JUDGE_IDS, seed=1)
    assert np.isfinite(scores.to_numpy(dtype=float)).all()

