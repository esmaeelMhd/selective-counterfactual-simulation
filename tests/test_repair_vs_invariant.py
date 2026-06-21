from __future__ import annotations

import pandas as pd

from scs.experiments.repair_signal_semantics import _signal_metrics_for_frame, repair_vs_invariant_summary


def test_signal_metrics_compute_near_zero_and_auroc() -> None:
    frame = pd.DataFrame(
        {
            "repair_amount": [0.0, 0.0, 1.0, 1.0],
            "rmse": [0.1, 0.2, 0.3, 0.4],
            "bad_rmse_label": [False, True, True, True],
            "accepted_region": ["accepted_good", "accepted_bad", "accepted_bad", "rejected_bad"],
        }
    )
    metrics = _signal_metrics_for_frame(frame, "repair_amount", "all", low_threshold=1e-9)
    assert metrics["near_zero_fraction"] == 0.5
    assert metrics["accepted_bad_low_signal_fraction"] == 0.5
    assert metrics["auroc_for_bad_rmse_label"] > 0.5


def test_repair_vs_invariant_verdict_rules_count_invariant_dominance() -> None:
    metrics = pd.DataFrame(
        [
            {"system_id": "cstr", "signal": "repair_amount", "accepted_region": "all", "auroc_for_bad_rmse_label": 0.5, "auprc_for_bad_rmse_label": 0.7, "spearman_correlation_with_rmse": 0.0, "near_zero_fraction": 1.0},
            {"system_id": "cstr", "signal": "invariant_residual", "accepted_region": "all", "auroc_for_bad_rmse_label": 0.91, "auprc_for_bad_rmse_label": 0.9, "spearman_correlation_with_rmse": 0.8, "near_zero_fraction": 0.0},
            {"system_id": "two_tank", "signal": "repair_amount", "accepted_region": "all", "auroc_for_bad_rmse_label": 0.5, "auprc_for_bad_rmse_label": 0.6, "spearman_correlation_with_rmse": 0.0, "near_zero_fraction": 1.0},
            {"system_id": "two_tank", "signal": "invariant_residual", "accepted_region": "all", "auroc_for_bad_rmse_label": 0.8, "auprc_for_bad_rmse_label": 0.8, "spearman_correlation_with_rmse": 0.6, "near_zero_fraction": 0.0},
        ]
    )
    joint = pd.DataFrame([{"low_repair_high_invariant_bad_count": 3}])
    config = {"diagnostic_thresholds": {"max_repair_auroc_blind": 0.60, "min_repair_auroc_useful": 0.70}}
    summary = repair_vs_invariant_summary(metrics, joint, config)
    assert summary["verdict"] == "INVARIANT_DOMINATES_REPAIR"
    assert summary["low_repair_high_invariant_bad_count_total"] == 3
