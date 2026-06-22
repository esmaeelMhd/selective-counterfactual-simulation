from __future__ import annotations

import json
from pathlib import Path


def test_benchmark_usability_package_check_is_accepted() -> None:
    result = json.loads(Path("results/benchmark_usability/package_check.json").read_text(encoding="utf-8"))

    assert result["verdict"] == "BENCHMARK_USABILITY_PACKAGE_ACCEPTED"
    assert result["demo_check"] == "DEMO_BUILT"
    assert result["model_comparison_check"] == "MODEL_COMPARISON_BUILT"
    assert result["custom_model_comparison_check"] == "MODEL_COMPARISON_BUILT"
    assert result["prior_artifact_mutation_detected"] is False
