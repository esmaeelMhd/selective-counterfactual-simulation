from __future__ import annotations

from pathlib import Path

from scs.experiments.current_status import _scan_claim_file


def test_claim_language_guard_flags_positive_overclaim(tmp_path: Path) -> None:
    path = tmp_path / "bad.md"
    path.write_text("# Results\n\nThis is strong support for high-coverage reliability.\n", encoding="utf-8")
    violations: list[dict] = []
    allowed: list[dict] = []

    _scan_claim_file(path, violations, allowed)

    assert any(item["phrase"] == "strong support" for item in violations)
    assert any(item["phrase"] == "high-coverage reliability" for item in violations)


def test_claim_language_guard_allows_forbidden_context(tmp_path: Path) -> None:
    path = tmp_path / "allowed.md"
    path.write_text(
        "# Current Evidence Status\n\n**What is not supported:**\n- safety certification\n- plant-wide digital twin claims\n",
        encoding="utf-8",
    )
    violations: list[dict] = []
    allowed: list[dict] = []

    _scan_claim_file(path, violations, allowed)

    assert violations == []
    assert {item["phrase"] for item in allowed} >= {"safety certification", "plant-wide digital twin"}
