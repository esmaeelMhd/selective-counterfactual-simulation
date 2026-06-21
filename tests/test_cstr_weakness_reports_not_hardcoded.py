from __future__ import annotations

from pathlib import Path

import pandas as pd

from scs.experiments.cstr_weakness import write_repair_report, write_signal_overlap_report


def test_repair_report_uses_supplied_numeric_values(tmp_path: Path) -> None:
    output = tmp_path / "repair.md"
    metrics = {
        "repair_auroc_for_bad_rmse_label": 0.123456,
        "fraction_accepted_false_accepts_with_low_repair": 0.75,
        "verdict": "REPAIR_SIGNAL_BLIND_SPOT",
        "key_finding": "fixture 0.123456",
    }
    distribution = pd.DataFrame([{"group": "accepted_bad", "mean": 1.0, "median": 1.0, "p10": 0.5, "p90": 1.5, "zero_fraction": 0.0}])
    low = pd.DataFrame([{"coverage": 0.05, "accepted_false_accept_count": 4, "low_repair_false_accept_count": 3, "fraction": 0.75}])
    comp = pd.DataFrame([{"coverage": 0.05, "repair_only_far": 0.9, "calibrated_far": 0.8}])
    write_repair_report(metrics, distribution, low, comp, output)
    text = output.read_text(encoding="utf-8")
    assert "0.123456" in text
    assert "REPAIR_SIGNAL_BLIND_SPOT" in text


def test_signal_overlap_report_changes_with_fixture_values(tmp_path: Path) -> None:
    output = tmp_path / "signal.md"
    metrics = pd.DataFrame([{"signal": "support_distance", "auroc": 0.654321, "cohens_d": 1.0, "overlap_coefficient": 0.2, "verdict": "MIXED"}])
    summary = {"key_finding": "fixture 0.654321", "verdict": "MIXED_SEPARABILITY"}
    write_signal_overlap_report(summary, metrics, output)
    assert "0.654321" in output.read_text(encoding="utf-8")
