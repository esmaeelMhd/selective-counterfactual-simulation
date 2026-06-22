from __future__ import annotations

from pathlib import Path

from scs.experiments.v2 import V2_REQUIRED_DECISIONS, decide_v2_claim, make_v2_scientific_decision_gate


def test_v2_decision_fixture_labels() -> None:
    heat = {"verdict": "VALID_HEAT_EXCHANGER_BENCHMARK"}
    base_run = {"valid_systems": ["two_tank", "cstr", "heat_exchanger"], "leakage_detected": False}
    strong = {
        "verdict": "STRONG_MULTI_SYSTEM_EFFECT",
        "positive_systems": ["two_tank", "cstr", "heat_exchanger"],
        "practical_threshold_systems": ["two_tank", "cstr"],
        "ci_positive_systems": ["two_tank", "cstr"],
        "event_risk_worsening": False,
        "leakage_detected": False,
    }
    assert decide_v2_claim(heat, base_run, strong)["decision"] == "UPGRADE_TO_MODERATE_MULTI_SYSTEM_LOW_COVERAGE_CLAIM"

    weak = {**strong, "positive_systems": ["two_tank", "cstr", "heat_exchanger"], "practical_threshold_systems": [], "ci_positive_systems": []}
    assert decide_v2_claim(heat, base_run, weak)["decision"] == "KEEP_WEAK_LOW_COVERAGE_BENCHMARK_CLAIM"

    mixed = {**strong, "positive_systems": ["two_tank"], "practical_threshold_systems": [], "ci_positive_systems": []}
    assert decide_v2_claim(heat, base_run, mixed)["decision"] == "SYSTEM_DEPENDENT_BENCHMARK_RESULT"

    none = {**strong, "positive_systems": [], "practical_threshold_systems": [], "ci_positive_systems": [], "verdict": "NO_ROBUST_EFFECT"}
    assert decide_v2_claim(heat, base_run, none)["decision"] == "NO_METHOD_CLAIM_BENCHMARK_ONLY"

    invalid = {**strong, "leakage_detected": True}
    assert decide_v2_claim(heat, base_run, invalid)["decision"] == "INVALID_V2_PROTOCOL"


def test_v2_decision_gate_report_uses_artifacts(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.md"
    protocol.write_text("# v2 Scientific Protocol Lock\n\n## Forbidden changes after results\n", encoding="utf-8")
    heat = tmp_path / "heat.json"
    frozen = tmp_path / "frozen.json"
    stats = tmp_path / "stats.json"
    heat.write_text('{"verdict":"VALID_HEAT_EXCHANGER_BENCHMARK"}', encoding="utf-8")
    frozen.write_text('{"valid_systems":["two_tank"],"models_evaluated":["hold_last"],"badness_targets":["bad_rmse"],"leakage_detected":false}', encoding="utf-8")
    stats.write_text('{"verdict":"NO_ROBUST_EFFECT","positive_systems":[],"practical_threshold_systems":[],"ci_positive_systems":[],"event_risk_worsening":false,"leakage_detected":false}', encoding="utf-8")
    output = tmp_path / "decision.md"
    result = make_v2_scientific_decision_gate(protocol, heat, frozen, stats, output)
    assert result["decision"] in V2_REQUIRED_DECISIONS
    assert result["decision"] in output.read_text(encoding="utf-8")
