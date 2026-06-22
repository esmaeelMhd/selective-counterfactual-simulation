from __future__ import annotations

from pathlib import Path

from scs.experiments.technical_note_package import build_limitations_first_technical_note, build_technical_note_evidence_tables
from test_technical_note_evidence_tables import _write_fixture_package


def test_generated_note_uses_fixture_table_values(tmp_path: Path) -> None:
    config = _write_fixture_package(tmp_path)
    tables = tmp_path / "tables"
    figures = tmp_path / "figures"
    figures.mkdir()
    for name in ["main_low_coverage_margins.png", "twotank_vs_cstr_far.png", "signal_role_summary.png", "smoke_model_sanity.png"]:
        (figures / name).write_bytes(b"fixture")
    build_technical_note_evidence_tables(config, tables, report_output=tmp_path / "tables.md")
    output = tmp_path / "note.md"
    build_limitations_first_technical_note(config, tables, figures, output)

    text = output.read_text(encoding="utf-8")
    assert "0.300000" in text
    assert "0.040000" in text
    assert "0.910000" in text


def test_evidence_table_report_uses_fixture_values(tmp_path: Path) -> None:
    config = _write_fixture_package(tmp_path)
    build_technical_note_evidence_tables(config, tmp_path / "tables", report_output=tmp_path / "report.md")

    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "0.300000" in text
    assert "0.040000" in text
