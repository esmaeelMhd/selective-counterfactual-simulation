from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scs.reports.summary import write_smoke_report


def test_report_changes_when_input_csv_changes(tmp_path) -> None:
    summary = {
        "config": {"experiment_id": "tmp"},
        "dataset_summary": {
            "id_test": {
                "system_id": "two_tank",
                "n_trajectories": 1,
                "horizon": 1,
                "state_dim": 2,
                "action_dim": 1,
                "disturbance_dim": 2,
                "action_min": 0.0,
                "action_max": 1.0,
                "disturbance_0_max": 1.0,
                "scenario_type": "normal_policy",
            }
        },
        "scenario_score_rows": 1,
        "combined_judge_result": {"statement": "generated statement"},
        "known_failures": [],
    }
    risk = pd.DataFrame(
        {
            "system_id": ["two_tank"],
            "model_id": ["linear_narx"],
            "split": ["id_test"],
            "judge_id": ["combined_linear"],
            "coverage": [0.5],
            "false_accept_rate": [0.123456],
            "accepted_count": [1],
            "false_accept_count": [0],
            "mean_error_accepted": [0.1],
            "mean_error_rejected": [0.0],
            "threshold": [0.15],
        }
    )
    metrics = pd.DataFrame(
        {
            "system_id": ["two_tank"],
            "model_id": ["linear_narx"],
            "split": ["id_test"],
            "rmse_mean": [0.1],
            "mae_mean": [0.1],
            "max_abs_error_mean": [0.1],
            "final_state_error_mean": [0.1],
        }
    )
    report_a = tmp_path / "a.md"
    write_smoke_report(summary, risk, metrics, report_a, "cmd")
    text_a = report_a.read_text()
    risk.loc[0, "false_accept_rate"] = 0.987654
    report_b = tmp_path / "b.md"
    write_smoke_report(summary, risk, metrics, report_b, "cmd")
    text_b = report_b.read_text()
    assert "0.123456" in text_a
    assert "0.987654" in text_b
    assert text_a != text_b

