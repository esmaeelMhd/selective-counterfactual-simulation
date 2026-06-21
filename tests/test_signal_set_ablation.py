from __future__ import annotations

import pandas as pd
import yaml

from scs.experiments.repair_signal_semantics import (
    BASE_SIGNALS,
    _signal_set_difference_vs_full,
    load_repair_signal_semantics_config,
    signal_set_ablation_summary,
)


def test_signal_set_ablation_summary_applies_no_repair_rules() -> None:
    low = pd.DataFrame(
        [
            {"system_id": "cstr", "signal_set_id": "full_original", "coverage": 0.05, "seed": 0, "absolute_margin": 0.01, "leakage_detected": False},
            {"system_id": "cstr", "signal_set_id": "no_repair", "coverage": 0.05, "seed": 0, "absolute_margin": 0.04, "leakage_detected": False},
            {"system_id": "two_tank", "signal_set_id": "full_original", "coverage": 0.05, "seed": 0, "absolute_margin": 0.10, "leakage_detected": False},
            {"system_id": "two_tank", "signal_set_id": "no_repair", "coverage": 0.05, "seed": 0, "absolute_margin": 0.09, "leakage_detected": False},
        ]
    )
    config = {"diagnostic_thresholds": {"min_cstr_absolute_margin_improvement": 0.02, "max_allowed_twotank_margin_drop": 0.02}}
    diff = _signal_set_difference_vs_full(low)
    summary = signal_set_ablation_summary(low, diff, config)
    assert summary["verdict"] == "NO_REPAIR_IMPROVES_CSTR_WITHOUT_HURTING_TWOTANK"


def test_signal_set_config_uses_only_existing_signals() -> None:
    config = load_repair_signal_semantics_config("configs/audits/repair_signal_semantics_audit.yaml")
    for signals in config["signal_sets"].values():
        assert set(signals) <= set(BASE_SIGNALS)
    raw = yaml.safe_load(open("configs/audits/repair_signal_semantics_audit.yaml", encoding="utf-8"))
    assert raw["forbidden"]["allow_new_judge_families"] is False
