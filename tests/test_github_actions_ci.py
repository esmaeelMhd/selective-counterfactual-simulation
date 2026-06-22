from __future__ import annotations

from pathlib import Path


def test_github_actions_ci_runs_public_reproducibility_paths() -> None:
    text = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "pip install -e \".[dev]\"" in text
    assert "pytest -q" in text
    assert "python scripts/run_smoke_demo.py --output results/ci_smoke_demo" in text
    assert "python examples/custom_model_example.py --output results/ci_custom_model_example" in text
    assert "python scripts/compare_models.py" in text
    assert "python scripts/update_readme_public_landing.py" in text
    assert "python scripts/check_claim_language.py" in text
    assert "secrets." not in text
