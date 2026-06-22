from __future__ import annotations

from pathlib import Path

import json

from scs.experiments.technical_note_package import REQUIRED_FIGURES, build_technical_note_evidence_tables, build_technical_note_figures
from test_technical_note_evidence_tables import _write_fixture_package


def test_technical_note_figures_are_created_from_tables(tmp_path: Path) -> None:
    config = _write_fixture_package(tmp_path)
    tables = tmp_path / "tables"
    figures = tmp_path / "figures"
    build_technical_note_evidence_tables(config, tables, report_output=tmp_path / "tables.md")
    build_technical_note_figures(config, tables, figures, report_output=tmp_path / "figures.md", manifest_output=tmp_path / "figure_manifest.json")

    for name in REQUIRED_FIGURES:
        path = figures / name
        assert path.exists(), path
        assert path.stat().st_size > 0, path
    manifest = json.loads((tmp_path / "figure_manifest.json").read_text(encoding="utf-8"))
    assert "main_result_table" in manifest["source_tables"]
    assert manifest["verdict"] == "FIGURES_BUILT"
