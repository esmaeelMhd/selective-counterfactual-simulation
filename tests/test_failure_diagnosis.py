from __future__ import annotations

import json
from pathlib import Path

import pytest

from scs.reports.failure_analysis import make_failure_diagnosis


def _write_diagnosis_inputs(path: Path, **overrides: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    data = {
        "gate_verification.json": {"verdict": "READY_FOR_FAILURE_ANALYSIS", "expansion_status": "BLOCKED"},
        "failure_table_schema.json": {"row_count": 10},
        "signal_error_correlation_summary.json": {"verdict": "NO_SIGNAL", "key_finding": "signals weak"},
        "per_split_summary.json": {"verdict": "GLOBAL_FAILURE", "key_finding": "all splits", "combined_failed_splits": []},
        "threshold_sensitivity_summary.json": {"verdict": "UNSUPPORTED_ACROSS_THRESHOLDS", "key_finding": "no thresholds"},
        "coverage_sensitivity_summary.json": {"verdict": "FAILS_ACROSS_COVERAGE", "key_finding": "no coverages"},
        "score_ablation_summary.json": {"verdict": "SIGNAL_PROBLEM", "key_finding": "no score helps"},
        "model_diversity_summary.json": {"verdict": "ORACLE_GAP_SMALL", "key_finding": "oracle small"},
        "benchmark_sanity_summary.json": {"verdict": "VALID_BENCHMARK", "key_finding": "benchmark valid"},
    }
    key_by_short = {
        "signal": "signal_error_correlation_summary.json",
        "ablation": "score_ablation_summary.json",
        "model": "model_diversity_summary.json",
        "benchmark": "benchmark_sanity_summary.json",
    }
    for key, value in overrides.items():
        data[key_by_short[key]]["verdict"] = value
    for filename, content in data.items():
        (path / filename).write_text(json.dumps(content), encoding="utf-8")


def test_failure_diagnosis_preserves_unsupported_claim(tmp_path: Path) -> None:
    _write_diagnosis_inputs(tmp_path)
    result = make_failure_diagnosis(tmp_path, tmp_path / "failure_diagnosis.md")
    assert result["diagnosis"] == "IDEA_NOT_SUPPORTED"
    assert result["recommended_next_action"] == "KILL_OR_DOWNGRADE_CLAIM"
    text = (tmp_path / "failure_diagnosis.md").read_text(encoding="utf-8")
    assert "Do not add CSTR, RSSM, or new systems" in text


def test_failure_diagnosis_raises_for_missing_inputs(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        make_failure_diagnosis(tmp_path, tmp_path / "failure_diagnosis.md")


def test_failure_diagnosis_classifies_useful_signals_bad_combined_as_judge_problem(tmp_path: Path) -> None:
    _write_diagnosis_inputs(tmp_path, signal="USEFUL_SIGNALS_FOUND")
    split = json.loads((tmp_path / "per_split_summary.json").read_text(encoding="utf-8"))
    split["combined_failed_splits"] = ["id_test"]
    (tmp_path / "per_split_summary.json").write_text(json.dumps(split), encoding="utf-8")
    result = make_failure_diagnosis(tmp_path, tmp_path / "failure_diagnosis.md")
    assert result["diagnosis"] == "JUDGE_PROBLEM"
    assert result["recommended_next_action"] == "REPLACE_JUDGE"
