from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scs.experiments.public_benchmark import (
    build_readme_main_figure,
    twotank_main_result_table,
    write_failure_gallery_markdown,
    write_twotank_reproduction_report,
)


def test_readme_figure_changes_with_temp_manifest(tmp_path: Path) -> None:
    manifest = json.loads(Path("results/current_status/evidence_manifest/current_evidence_manifest.json").read_text(encoding="utf-8"))
    manifest["systems"]["two_tank"]["coverage_0_05_margin"] = 0.321
    manifest["systems"]["cstr"]["coverage_0_05_margin"] = 0.012
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    report = tmp_path / "report.md"

    result = build_readme_main_figure(manifest_path, tmp_path / "figure.png", report_output=report)

    assert any(row["margin"] == 0.321 for row in result["rows"])
    assert "0.321000" in report.read_text(encoding="utf-8")


def test_reproduction_report_changes_with_temp_source_table(tmp_path: Path) -> None:
    source = tmp_path / "low_coverage_summary.csv"
    pd.DataFrame(
        [
            {"coverage": 0.05, "baseline_far": 0.9, "calibrated_far": 0.5, "margin": 0.4},
            {"coverage": 0.10, "baseline_far": 0.8, "calibrated_far": 0.7, "margin": 0.1},
        ]
    ).to_csv(source, index=False)
    table = twotank_main_result_table(source)
    summary = {
        "source_artifact": str(source),
        "coverage_0_05_margin": 0.4,
        "coverage_0_10_margin": 0.1,
        "verdict": "FIXTURE",
    }
    output = tmp_path / "report.md"

    write_twotank_reproduction_report(table, summary, output)

    text = output.read_text(encoding="utf-8")
    assert "0.400000" in text
    assert str(source) in text


def test_failure_gallery_markdown_changes_with_example_fixture(tmp_path: Path) -> None:
    example = {
        "example_id": "example_1_accepted_good",
        "title": "Example 1: Accepted good scenario",
        "system": "fixture_system",
        "model": "fixture_model",
        "scenario_type": "fixture_scenario",
        "judge": "fixture_judge",
        "coverage": 0.05,
        "rmse": 0.123,
        "decision": "accepted",
        "false_accept": False,
        "source_artifact": "fixture.csv",
        "figure": "fixture.png",
        "key_signal_values": {
            "support_distance": 1.0,
            "uncertainty_score": 2.0,
            "disagreement_score": 3.0,
            "invariant_residual": 4.0,
            "repair_amount": 0.0,
            "risk_value": 5.0,
        },
    }
    examples = [
        example,
        {**example, "example_id": "example_2_correctly_rejected_bad", "title": "Example 2: Correctly rejected bad scenario", "decision": "refused"},
        {**example, "example_id": "example_3_false_accept_cstr", "title": "Example 3: False accept", "false_accept": True},
        {**example, "example_id": "example_4_cstr_within_bound_dynamic_failure", "title": "Example 4: CSTR within-bound dynamic failure"},
        {**example, "example_id": "example_5_invariant_residual_helps", "title": "Example 5: Invariant residual helps"},
    ]
    output = tmp_path / "gallery.md"

    write_failure_gallery_markdown(examples, output)

    text = output.read_text(encoding="utf-8")
    assert "fixture_system" in text
    assert "support=1.000000" in text
