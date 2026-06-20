from __future__ import annotations

from pathlib import Path

from conftest import write_failure_results_fixture
from scs.reports.failure_analysis import build_failure_table


def test_failure_table_report_changes_when_input_changes(tmp_path: Path) -> None:
    one = write_failure_results_fixture(tmp_path / "one", n_per_split=4)
    out_one = tmp_path / "out_one" / "failure_table.csv"
    schema_one = build_failure_table(one, out_one)
    text_one = Path("reports/failure_table_report.md").read_text(encoding="utf-8")

    two = write_failure_results_fixture(tmp_path / "two", n_per_split=5)
    out_two = tmp_path / "out_two" / "failure_table.csv"
    schema_two = build_failure_table(two, out_two)
    text_two = Path("reports/failure_table_report.md").read_text(encoding="utf-8")

    assert str(schema_one["row_count"]) in text_one
    assert str(schema_two["row_count"]) in text_two
    assert schema_one["row_count"] != schema_two["row_count"]
    assert text_one != text_two
