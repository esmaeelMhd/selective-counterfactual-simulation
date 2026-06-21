from __future__ import annotations

import numpy as np
import pandas as pd

from scs.experiments.effect_audit import _bootstrap_margin_samples, _ci, _paired_seed_ci


def _fixture_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model_id": ["m"] * 4,
            "scenario_type": ["s"] * 4,
            "bad_rmse_label": [False, True, True, False],
            "risk_best_single_signal_selected_on_calibration": [0.1, 0.2, 0.3, 0.4],
            "risk_rank_normalized_linear": [0.1, 0.4, 0.5, 0.2],
            "risk_calibration_selected_candidate_ranker": [0.2, 0.3, 0.4, 0.1],
        }
    )


def test_bootstrap_ci_is_finite_reproducible_and_contains_mean() -> None:
    table = _fixture_table()
    samples_a = _bootstrap_margin_samples(
        table,
        "best_single_signal_selected_on_calibration",
        ["rank_normalized_linear", "calibration_selected_candidate_ranker"],
        0.5,
        iterations=30,
        seed=123,
    )
    samples_b = _bootstrap_margin_samples(
        table,
        "best_single_signal_selected_on_calibration",
        ["rank_normalized_linear", "calibration_selected_candidate_ranker"],
        0.5,
        iterations=30,
        seed=123,
    )
    assert samples_a["absolute_margin"].equals(samples_b["absolute_margin"])
    low, high = _ci(samples_a["absolute_margin"], 0.95)
    mean = float(samples_a["absolute_margin"].mean())
    assert np.isfinite([low, high]).all()
    assert low <= mean <= high


def test_degenerate_seed_ci_returns_degenerate_bounds() -> None:
    low, high = _paired_seed_ci(np.array([0.1, 0.1, 0.1]), 0.95)
    assert low == high == 0.1
