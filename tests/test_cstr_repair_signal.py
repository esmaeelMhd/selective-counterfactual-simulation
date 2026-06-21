from __future__ import annotations

import pandas as pd

from scs.experiments.cstr_weakness import _auroc, _repair_metrics_verdict, _spearman


def test_repair_auroc_and_correlation_calculations() -> None:
    labels = pd.Series([False, False, True, True])
    scores = pd.Series([0.1, 0.2, 0.8, 0.9])
    assert _auroc(labels, scores) == 1.0
    assert _spearman(scores, pd.Series([0.1, 0.2, 0.8, 0.9])) == 1.0


def test_zero_and_low_repair_blindspot_verdict() -> None:
    metrics = {
        "repair_auroc_for_bad_rmse_label": 0.5,
        "fraction_accepted_false_accepts_with_low_repair": 1.0,
        "repair_dynamic_range": 0.0,
        "out_of_bounds_or_repair_events_observed": False,
    }
    assert _repair_metrics_verdict(metrics) == "REPAIR_SIGNAL_BLIND_SPOT"


def test_miscomputed_verdict_requires_repair_events() -> None:
    metrics = {
        "repair_auroc_for_bad_rmse_label": 0.5,
        "fraction_accepted_false_accepts_with_low_repair": 1.0,
        "repair_dynamic_range": 0.0,
        "out_of_bounds_or_repair_events_observed": True,
    }
    assert _repair_metrics_verdict(metrics) == "REPAIR_SIGNAL_MISCOMPUTED"
