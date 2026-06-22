from __future__ import annotations

import json
from pathlib import Path


def test_public_benchmark_package_check_is_accepted() -> None:
    result = json.loads(Path("results/public_benchmark_v1_2/package_check.json").read_text(encoding="utf-8"))

    assert result["verdict"] == "PUBLIC_BENCHMARK_PACKAGE_ACCEPTED"
    assert result["smoke_demo"] == "SMOKE_DEMO_BUILT"
    assert result["twotank_reproduction"] == "TWOTANK_MAIN_RESULT_REPRODUCED"
    assert result["prior_artifact_mutation_detected"] is False
    assert result["claim_language_violations"] == []
