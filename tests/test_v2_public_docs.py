from __future__ import annotations

from pathlib import Path


def test_readme_and_public_docs_frame_benchmark() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    assert readme.startswith("# Selective Counterfactual Simulation Benchmark")
    assert "A benchmark for testing whether learned dynamical simulators know when to refuse counterfactual predictions." in readme
    assert "calibrated refusal is target-dependent and not reliable for event-risk" in readme
    assert "python scripts/run_benchmark.py --model examples/custom_model_example.py:DampedLinearUserModel" in readme
    task = Path("docs/v2/benchmark_task.md").read_text(encoding="utf-8")
    assert "## False accept definition" in task
    assert "## Coverage definition" in task
    assert "## Event-risk caveat" in task
    card = Path("docs/v2/benchmark_card.md").read_text(encoding="utf-8")
    assert "## Non-intended use" in card
    assert "safety certification" in card
