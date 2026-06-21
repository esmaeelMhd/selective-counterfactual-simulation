from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from conftest import write_tiny_calibrated_cstr_config
from scs.experiments.calibrated import load_calibrated_config, run_calibrated_judge, write_calibrated_report
from scs.experiments.cstr_replication import write_cstr_sanity_report


def test_calibrated_cstr_report_values_come_from_tables(tmp_path: Path) -> None:
    config_path = write_tiny_calibrated_cstr_config(tmp_path / "tiny_cstr.yaml")
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
    low.loc[0, "margin"] = 7.654321
    write_calibrated_report(config, summary, risk, comparison, low, selection, tmp_path / "report_b.md", "cmd")
    assert "7.654321" in (tmp_path / "report_b.md").read_text(encoding="utf-8")
    assert text_a != (tmp_path / "report_b.md").read_text(encoding="utf-8")


def test_cstr_sanity_report_values_come_from_payload(tmp_path: Path) -> None:
    config = {"experiment_id": "x", "seed": 1, "horizon": 2, "dt": 0.1}
    data_summary = {
        "finite": True,
        "nonconstant": True,
        "physically_plausible": True,
        "scenario_types": ["id"],
    }
    distribution = {"cooling_action_shift": 2.0, "feed_concentration_shift": 2.0, "feed_temperature_shift": 20.0}
    model_errors = pd.DataFrame({"model": ["m"], "id_rmse": [0.1], "ood_rmse": [0.2], "ood_minus_id": [0.1], "passed": [True]})
    labels = {
        "roles": [{"role": "judge_test", "row_count": 2, "bad_count": 1, "bad_rate": 0.5, "non_degenerate": True}],
        "split_overlap": {"scenario_overlap_count": 0},
        "verdict": "VALID_CSTR_BENCHMARK",
        "invalid_reasons": [],
        "weak_reasons": [],
    }
    write_cstr_sanity_report(config, data_summary, distribution, model_errors, labels, tmp_path / "sanity.md")
    text = (tmp_path / "sanity.md").read_text(encoding="utf-8")
    assert "0.200000" in text
    model_errors.loc[0, "ood_rmse"] = 9.9
    write_cstr_sanity_report(config, data_summary, distribution, model_errors, labels, tmp_path / "sanity_b.md")
    assert "9.900000" in (tmp_path / "sanity_b.md").read_text(encoding="utf-8")
