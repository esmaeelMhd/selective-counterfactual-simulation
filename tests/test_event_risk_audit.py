from __future__ import annotations

from pathlib import Path

import pandas as pd

import scs.experiments.effect_audit as effect_audit
from scs.experiments.effect_audit import _event_flags, _event_risk_rows, analyze_event_risk


def test_event_labels_are_computed_from_states() -> None:
    states = [[1.0, 340.0], [0.2, 391.0]]
    flags = _event_flags(
        "cstr",
        states,
        {"temperature_high": 390.0, "concentration_low": 0.25, "concentration_high": 1.8},
    )
    assert flags["temperature_above_limit"] is True
    assert flags["concentration_out_of_safe_range"] is True
    assert flags["unsafe_reactor_state"] is True


def test_event_far_is_computed_from_event_labels() -> None:
    merged = pd.DataFrame(
        {
            "system_id": ["cstr", "cstr"],
            "model_id": ["m", "m"],
            "scenario_type": ["s", "s"],
            "bad_rmse_label": [False, False],
            "bad_event": [True, False],
            "true_any_event": [True, False],
            "predicted_any_event": [False, False],
            "risk_best_single_signal_selected_on_calibration": [0.1, 0.2],
        }
    )
    rows = _event_risk_rows(merged, ["best_single_signal_selected_on_calibration"], [0.5], ["bad_event"])
    assert rows.iloc[0]["false_accept_rate"] == 1.0
    assert rows.iloc[0]["accepted_event_bad_count"] == 1


def test_missing_event_trajectory_data_is_reported_unavailable(monkeypatch, tmp_path: Path) -> None:
    def raise_missing(system_id, event_config):
        raise FileNotFoundError("missing trajectory data")

    monkeypatch.setattr(effect_audit, "_event_label_table", raise_missing)
    summary = analyze_event_risk(
        "configs/audits/effect_size_audit.yaml",
        "configs/audits/event_risk_audit.yaml",
        tmp_path / "event",
        report_path=tmp_path / "event_report.md",
    )
    assert summary["verdict"] == "EVENT_UNAVAILABLE"
    assert "missing trajectory data" in summary["event_label_availability"][0]["missing_reason"]
