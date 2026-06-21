from __future__ import annotations

import pandas as pd

from scs.experiments.repair_signal_semantics import signal_set_seed_sweep_summary


def test_seed_sweep_summary_includes_two_seed_fixture_and_both_systems() -> None:
    rows = []
    for seed in [0, 1]:
        rows.extend(
            [
                {"system_id": "cstr", "signal_set_id": "full_original", "coverage": 0.05, "seed": seed, "absolute_margin": 0.01, "run_status": "passed"},
                {"system_id": "cstr", "signal_set_id": "no_repair", "coverage": 0.05, "seed": seed, "absolute_margin": 0.04, "run_status": "passed"},
                {"system_id": "two_tank", "signal_set_id": "full_original", "coverage": 0.05, "seed": seed, "absolute_margin": 0.10, "run_status": "passed"},
                {"system_id": "two_tank", "signal_set_id": "no_repair", "coverage": 0.05, "seed": seed, "absolute_margin": 0.09, "run_status": "passed"},
            ]
        )
    config = {"diagnostic_thresholds": {"min_cstr_absolute_margin_improvement": 0.02, "max_allowed_twotank_margin_drop": 0.02}}
    summary = signal_set_seed_sweep_summary(pd.DataFrame(rows), [], config, [0, 1])
    assert summary["seeds"] == [0, 1]
    assert summary["cstr_improve_seed_count"] == 2
    assert summary["twotank_harm_seed_count"] == 0
    assert summary["failed_seeds"] == []
