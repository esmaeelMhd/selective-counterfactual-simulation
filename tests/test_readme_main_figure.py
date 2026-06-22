from __future__ import annotations

import json
from pathlib import Path

from scs.experiments.public_benchmark import build_readme_main_figure


def test_readme_main_figure_builds_from_manifest(tmp_path: Path) -> None:
    output = tmp_path / "figure.png"
    report = tmp_path / "figure_report.md"

    manifest = build_readme_main_figure(
        "results/current_status/evidence_manifest/current_evidence_manifest.json",
        output,
        report_output=report,
    )

    assert output.exists()
    assert output.stat().st_size > 0
    assert manifest["source_manifest"] == "results/current_status/evidence_manifest/current_evidence_manifest.json"
    assert "not safety evidence" in manifest["subtitle"]
    assert "Low-coverage false-accept reduction" in report.read_text(encoding="utf-8")


def test_committed_readme_figure_manifest_is_public_safe() -> None:
    manifest = json.loads(Path("results/public_benchmark_v1_2/readme_figure_manifest.json").read_text(encoding="utf-8"))

    assert manifest["verdict"] == "README_MAIN_FIGURE_BUILT"
    assert manifest["source_manifest"] == "results/current_status/evidence_manifest/current_evidence_manifest.json"
    assert "strong" not in manifest["subtitle"].lower()
