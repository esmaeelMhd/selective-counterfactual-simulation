from __future__ import annotations

from pathlib import Path

import pytest

from scs.experiments.public_benchmark import update_readme_public_landing

def test_readme_starts_with_public_hook_and_required_sections() -> None:
    text = Path("README.md").read_text(encoding="utf-8")

    expected_opening = (
        "# Selective Counterfactual Simulation Benchmark\n\n"
        "A benchmark for testing whether learned dynamical simulators know when to refuse counterfactual predictions.\n\n"
        "Plug in a simulator, run OOD/intervention scenarios, and compare false-accept rate versus coverage.\n\n"
        "**Current status:** this is a benchmark prototype with narrow synthetic evidence."
    )
    assert text.startswith(expected_opening)
    for section in [
        "## What this is",
        "## Quickstart",
        "## Run the benchmark/demo",
        "## Plug in your own model",
        "## Current evidence",
        "## What is not claimed",
        "## Repository structure",
        "## Reproducibility",
        "## Citation",
        "## License",
    ]:
        assert section in text
    assert "python scripts/run_smoke.py" in text
    assert "python scripts/run_current_status_demo.py" in text
    assert "examples/custom_model_example.py:DampedLinearUserModel" in text
    assert "python scripts/run_benchmark.py --model" in text
    assert "docs/v2/figures/event_risk_vs_rmse_public.png" in text
    assert "event-risk remains a failure mode" in text
    assert "not a robust calibrated-refusal method claim" in text
    assert "This is not safety certification." in text


def test_readme_public_landing_check_catches_stale_readme(tmp_path: Path) -> None:
    stale = tmp_path / "README.md"
    stale.write_text("# Temporary\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="stale"):
        update_readme_public_landing("configs/status/public_benchmark_v1_2.yaml", stale, check=True)
