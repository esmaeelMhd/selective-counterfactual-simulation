from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from scs.experiments.v2_event_risk import event_guarded_score, load_event_risk_fix_config


def test_event_guarded_score_uses_existing_signal_ranks_not_labels() -> None:
    calibration = pd.DataFrame(
        {
            "risk_invariant_only": [0.0, 0.2, 0.8, 1.0],
            "risk_disagreement_only": [0.0, 0.1, 0.9, 1.0],
            "risk_support_only": [0.1, 0.2, 0.7, 0.8],
            "bad_label": [False, False, True, True],
        }
    )
    table = pd.DataFrame(
        {
            "risk_invariant_only": [0.1, 0.9],
            "risk_disagreement_only": [0.1, 0.95],
            "risk_support_only": [0.2, 0.75],
            "bad_label": [True, False],
        }
    )
    inputs = ["risk_invariant_only", "risk_disagreement_only", "risk_support_only"]
    score = event_guarded_score(calibration, table, inputs)
    flipped = event_guarded_score(
        calibration.assign(bad_label=~calibration["bad_label"]),
        table.assign(bad_label=~table["bad_label"]),
        inputs,
    )
    assert np.allclose(score, flipped)
    assert score[0] < score[1]


def test_event_risk_fix_config_forbids_expansion_and_claim_upgrade() -> None:
    config = load_event_risk_fix_config("configs/v2/v2_event_risk_fix.yaml")
    assert config["rules"]["forbid_new_systems"] is True
    assert config["rules"]["forbid_new_models"] is True
    assert config["rules"]["forbid_new_raw_signals"] is True
    assert config["rules"]["forbid_claim_upgrade"] is True


def test_event_risk_fix_outputs_improve_non_degenerate_event_systems() -> None:
    summary_path = Path("results/v2_event_risk_fix/event_risk_fix_summary.json")
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["verdict"] == "EVENT_GUARD_REDUCES_EVENT_FALSE_ACCEPTS_NO_CLAIM_UPGRADE"
    assert summary["scientific_claim_upgraded"] is False
    assert summary["uses_test_labels_for_scoring"] is False
    assert sorted(summary["nondegenerate_event_systems"]) == ["cstr", "heat_exchanger"]
    assert sorted(summary["improved_vs_primary_systems"]) == ["cstr", "heat_exchanger"]
    assert sorted(summary["nonworse_vs_fair_baseline_systems"]) == ["cstr", "heat_exchanger"]
    assert summary["beats_row_wise_envelope_systems"] == []


def test_event_risk_fix_comparison_columns_and_no_leakage_flags() -> None:
    comparison = pd.read_csv("results/v2_event_risk_fix/event_guarded_comparison.csv")
    required = {
        "system_id",
        "seed",
        "model_id",
        "badness_target",
        "coverage",
        "event_guard_far",
        "primary_far",
        "fair_baseline_far",
        "row_wise_envelope_far",
        "improvement_vs_primary",
        "margin_vs_fair_baseline",
        "margin_vs_row_wise_envelope",
        "uses_test_labels_for_scoring",
        "calibration_only_normalization",
    }
    assert required.issubset(comparison.columns)
    assert set(comparison["badness_target"]) == {"bad_event"}
    assert not comparison["uses_test_labels_for_scoring"].any()
    assert comparison["calibration_only_normalization"].all()
    by_system = comparison.groupby("system_id", as_index=False).agg(
        improvement_vs_primary=("improvement_vs_primary", "mean"),
        margin_vs_fair_baseline=("margin_vs_fair_baseline", "mean"),
    )
    nondegenerate = by_system[by_system["system_id"].isin(["cstr", "heat_exchanger"])]
    assert (nondegenerate["improvement_vs_primary"] > 0.0).all()
    assert (nondegenerate["margin_vs_fair_baseline"] >= 0.0).all()


def test_event_risk_fix_reports_do_not_claim_event_risk_solved() -> None:
    report = Path("reports/v2_event_risk_failure_diagnosis.md").read_text(encoding="utf-8").lower()
    decision = Path("reports/v2_event_risk_fix_decision.md").read_text(encoding="utf-8").lower()
    assert "does not support a general calibrated-refusal claim" in report
    assert "event-risk is solved" in decision
    assert "event-risk is solved." not in report
    assert "safety certification" in decision
