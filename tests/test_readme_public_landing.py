from __future__ import annotations

from pathlib import Path

import pytest

from scs.experiments.public_benchmark import update_readme_public_landing
from scs.experiments.v2_public_hardening import PUBLIC_HOOK


def test_readme_starts_with_public_hook_and_required_sections() -> None:
    text = Path("README.md").read_text(encoding="utf-8")

    expected_opening = (
        "# Selective Counterfactual Simulation Benchmark\n\n"
        f"{PUBLIC_HOOK}\n\n"
        "Plug in a simulator, run OOD/intervention scenarios, and compare false-accept rate versus coverage.\n\n"
        "**Current v2 finding:** calibrated refusal is target-dependent and not reliable for event-risk."
    )
    assert text.startswith(expected_opening)
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
    assert "examples/custom_model_example.py:DampedLinearUserModel" in text
    assert "python scripts/run_benchmark.py --model" in text
    assert "docs/v2/figures/event_risk_vs_rmse_public.png" in text
    assert "target-dependent calibrated-refusal failure" in text
    assert "does not support a robust method claim" in text
    assert "does not claim simulator safety" in text


def test_readme_public_landing_check_catches_stale_readme(tmp_path: Path) -> None:
    stale = tmp_path / "README.md"
    stale.write_text("# Temporary\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="stale"):
        update_readme_public_landing("configs/status/public_benchmark_v1_2.yaml", stale, check=True)
