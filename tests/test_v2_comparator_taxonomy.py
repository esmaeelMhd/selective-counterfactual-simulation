from __future__ import annotations

from pathlib import Path


def test_v2_comparator_taxonomy_required_sections_and_semantics() -> None:
    text = Path("docs/v2/v2_comparator_taxonomy.md").read_text(encoding="utf-8")
    required = [
        "# v2 Comparator Taxonomy",
        "## Purpose",
        "## Comparator 1: row-wise strongest-baseline envelope",
        "## Comparator 2: global calibration-selected fixed baseline",
        "## Comparator 3: per-system calibration-selected fixed baseline",
        "## Comparator 4: per-system-target calibration-selected fixed baseline",
        "## Comparator 5: current primary calibrated judge",
        "## Comparator 6: best calibrated-family member selected on calibration",
        "## Deployable vs diagnostic comparators",
        "## What uses test labels",
        "## What does not use test labels",
        "## Allowed interpretation",
        "## Forbidden interpretation",
    ]
    for section in required:
        assert section in text
    row_wise_section = text.split("## Comparator 1: row-wise strongest-baseline envelope", 1)[1].split("## Comparator 2", 1)[0]
    assert "diagnostic-only" in row_wise_section
    assert "not a deployable baseline" in row_wise_section
    assert "uses test outcomes row-wise" in row_wise_section
