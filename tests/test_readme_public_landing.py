from __future__ import annotations

from pathlib import Path

import pytest

from scs.experiments.public_benchmark import README_OPENING, update_readme_public_landing


def test_readme_starts_with_public_hook_and_required_sections() -> None:
    text = Path("README.md").read_text(encoding="utf-8")

    assert text.startswith(README_OPENING)
    for section in [
        "## Why this exists",
        "## Quickstart",
        "## Reproduce the main TwoTank result",
        "## Plug in your own simulator",
        "## Main result",
        "## What this does not claim",
        "## Repository map",
    ]:
        assert section in text
    assert "python scripts/reproduce_main_twotank_result.py --output results/reproduce_twotank" in text
    assert "examples/my_model_template.py" in text
    assert "docs/figures/readme_low_coverage_result.png" in text
    assert "weak on CSTR" in text
    assert "Not a safety tool" in text
    assert "does not claim simulator safety" in text


def test_readme_public_landing_check_catches_stale_readme(tmp_path: Path) -> None:
    stale = tmp_path / "README.md"
    stale.write_text("# Temporary\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="stale"):
        update_readme_public_landing("configs/status/public_benchmark_v1_2.yaml", stale, check=True)
