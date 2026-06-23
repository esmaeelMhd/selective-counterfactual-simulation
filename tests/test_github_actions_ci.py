from __future__ import annotations

from pathlib import Path


def test_github_actions_ci_runs_public_reproducibility_paths() -> None:
    text = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "pip install -e \".[dev]\"" in text
    assert "pytest -q" in text
    assert "python scripts/run_benchmark.py --model examples/custom_model_example.py:DampedLinearUserModel --output results/ci_public_benchmark" in text
    assert "python scripts/run_benchmark.py --models linear_narx mlp_state_space --output results/ci_builtin_benchmark" in text
    assert "python scripts/v2_build_public_event_risk_figure.py" in text
    assert "python scripts/v2_check_public_benchmark_package.py" in text
    assert "secrets." not in text
