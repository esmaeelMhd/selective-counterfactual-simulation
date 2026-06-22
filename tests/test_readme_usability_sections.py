from __future__ import annotations

from pathlib import Path

import pytest

from scs.experiments.benchmark_usability import REQUIRED_README_SECTIONS, update_readme_usability_sections


def test_readme_contains_required_usability_sections() -> None:
    text = Path("README.md").read_text(encoding="utf-8")

    for section in REQUIRED_README_SECTIONS:
        assert section in text
    assert "python scripts/run_current_status_demo.py" in text
    assert "examples/custom_model_example.py:DampedLinearUserModel" in text
    assert "## Claim Boundaries" in text


def test_readme_check_detects_stale_readme(tmp_path: Path) -> None:
    stale = tmp_path / "README.md"
    stale.write_text("# Temporary README\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="stale"):
        update_readme_usability_sections("configs/status/benchmark_usability_v1_1.yaml", stale, check=True)
