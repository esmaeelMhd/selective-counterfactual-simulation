from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from conftest import write_tiny_calibrated_config
from scs.experiments.calibrated import load_calibrated_config, run_calibrated_judge, write_calibrated_report


def test_calibrated_report_values_come_from_csv_json(tmp_path: Path) -> None:
    config_path = write_tiny_calibrated_config(tmp_path / "tiny.yaml")
    output = tmp_path / "run"
    run_calibrated_judge(config_path, output, report_path=tmp_path / "report_a.md")
    summary = json.loads((output / "calibrated_judge_summary.json").read_text(encoding="utf-8"))
    text_a = (tmp_path / "report_a.md").read_text(encoding="utf-8")
    assert summary["verdict"] in text_a

    config = load_calibrated_config(config_path)
    risk = pd.read_csv(output / "calibrated_risk_coverage.csv")
    comparison = pd.read_csv(output / "test_comparison.csv")
    low = pd.read_csv(output / "low_coverage_summary.csv")
    selection = pd.read_csv(output / "calibration_selection.csv")
    low.loc[0, "margin"] = 9.876543
    write_calibrated_report(config, summary, risk, comparison, low, selection, tmp_path / "report_b.md", "cmd")
    text_b = (tmp_path / "report_b.md").read_text(encoding="utf-8")
    assert "9.876543" in text_b
    assert text_a != text_b
