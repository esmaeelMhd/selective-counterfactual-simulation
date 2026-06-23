from __future__ import annotations

from pathlib import Path

from scs.experiments.v2 import make_v2_scientific_decision_gate


def test_v2_decision_report_changes_when_artifact_changes(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.md"
    protocol.write_text("# v2 Scientific Protocol Lock\n\n## Forbidden changes after results\n", encoding="utf-8")
    heat = tmp_path / "heat.json"
    frozen = tmp_path / "frozen.json"
    stats = tmp_path / "stats.json"
    heat.write_text('{"verdict":"VALID_HEAT_EXCHANGER_BENCHMARK"}', encoding="utf-8")
    frozen.write_text(
        '{"valid_systems":["two_tank","cstr","heat_exchanger"],"models_evaluated":["hold_last"],"badness_targets":["bad_rmse"],"leakage_detected":false}',
        encoding="utf-8",
    )
    stats.write_text(
        '{"verdict":"NO_ROBUST_EFFECT","positive_systems":[],"practical_threshold_systems":[],"ci_positive_systems":[],"event_risk_worsening":false,"leakage_detected":false}',
        encoding="utf-8",
    )
    output = tmp_path / "decision.md"
    make_v2_scientific_decision_gate(protocol, heat, frozen, stats, output)
    first = output.read_text(encoding="utf-8")
    stats.write_text(
        '{"verdict":"STRONG_MULTI_SYSTEM_EFFECT","positive_systems":["two_tank","cstr","heat_exchanger"],"practical_threshold_systems":["two_tank","cstr"],"ci_positive_systems":["two_tank","cstr"],"event_risk_worsening":false,"leakage_detected":false}',
        encoding="utf-8",
    )
    make_v2_scientific_decision_gate(protocol, heat, frozen, stats, output)
    second = output.read_text(encoding="utf-8")
    assert first != second
    assert "NO_METHOD_CLAIM_BENCHMARK_ONLY" in first
    assert "UPGRADE_TO_MODERATE_MULTI_SYSTEM_LOW_COVERAGE_CLAIM" in second
