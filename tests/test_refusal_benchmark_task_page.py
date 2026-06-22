from __future__ import annotations

from pathlib import Path


def test_refusal_benchmark_task_page_defines_task_and_non_goals() -> None:
    text = Path("docs/tasks/refusal_benchmark_task.md").read_text(encoding="utf-8")

    for section in [
        "# Task: Refusal Benchmark for Counterfactual Simulators",
        "## Task summary",
        "## Input",
        "## Output",
        "## Systems",
        "## Scenario types",
        "## Models",
        "## Refusal signals",
        "## Primary metric",
        "## False accept definition",
        "## Coverage definition",
        "## Baselines",
        "## Current best known result",
        "## How to submit/evaluate your own model locally",
        "## Fair comparison rules",
        "## Non-goals",
    ]:
        assert section in text
    assert "A false accept occurs when a judge accepts a scenario but the simulator rollout is bad under the configured error/event label." in text
    assert "Coverage is the fraction of scenarios accepted by the judge." in text
    assert "Weak-positive at low coverage; TwoTank stronger than CSTR." in text
    assert "not production validation" in text
