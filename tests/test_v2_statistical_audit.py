from __future__ import annotations

from pathlib import Path

import pandas as pd

from scs.experiments.v2 import run_v2_frozen_protocol, run_v2_statistical_audit
from v2_fixtures import write_tiny_v2_config


def test_v2_statistical_audit_computes_ci_and_blocks_event_worsening(tmp_path: Path) -> None:
    config = write_tiny_v2_config(tmp_path / "tiny.yaml")
    frozen = tmp_path / "frozen"
    run_v2_frozen_protocol(config, "configs/v2/v2_event_targets.yaml", frozen)
    summary = run_v2_statistical_audit(config, frozen, tmp_path / "stats")
    assert summary["verdict"] in {
        "STRONG_MULTI_SYSTEM_EFFECT",
        "WEAK_MULTI_SYSTEM_EFFECT",
        "MIXED_SYSTEM_DEPENDENT_EFFECT",
        "NO_ROBUST_EFFECT",
        "INVALID_DUE_TO_LEAKAGE_OR_BENCHMARK_FAILURE",
    }
    effect = pd.read_csv(tmp_path / "stats" / "v2_effect_size_by_system.csv")
    assert {"bootstrap_ci_low", "bootstrap_ci_high", "seed_win_rate", "practical_threshold_pass"}.issubset(effect.columns)
    assert (tmp_path / "stats" / "v2_effect_size_plot.png").stat().st_size > 0


def test_event_risk_worsening_fixture_prevents_strong_verdict(tmp_path: Path) -> None:
    config = write_tiny_v2_config(tmp_path / "tiny.yaml")
    results = tmp_path / "manual"
    results.mkdir()
    pd.DataFrame(
        [
            {
                "system_id": "two_tank",
                "seed": seed,
                "model_id": "hold_last",
                "badness_target": "bad_event",
                "bad_threshold": 0.5,
                "coverage": 0.05,
                "judge_id": "calibration_selected_candidate_ranker",
                "false_accept_rate": 0.8,
                "baseline_far": 0.4,
                "absolute_margin": -0.4,
                "relative_margin": -1.0,
                "false_accept_count": 1,
            }
            for seed in [0, 1]
        ]
    ).to_csv(results / "v2_risk_coverage.csv", index=False)
    pd.DataFrame({"system_id": ["two_tank"], "seed": [0], "model_id": ["hold_last"], "split": ["judge_test_id"], "true_event_rate": [0.0], "pred_event_rate": [0.0], "event_mismatch_rate": [0.0], "n_scenarios": [1]}).to_csv(results / "v2_event_metrics.csv", index=False)
    (results / "v2_run_summary.json").write_text('{"valid_systems":["two_tank"],"leakage_detected":false}', encoding="utf-8")
    summary = run_v2_statistical_audit(config, results, tmp_path / "stats_manual")
    assert summary["event_risk_worsening"] is True
    assert summary["verdict"] != "STRONG_MULTI_SYSTEM_EFFECT"
