from __future__ import annotations

from pathlib import Path

from scs.experiments.public_release import (
    QUICKSTART_COMMANDS,
    check_public_release_ready,
    scan_claim_language,
    scan_large_files,
    validate_readme,
)


def test_public_release_readme_and_docs_exist() -> None:
    for path in [
        "LICENSE",
        "CITATION.cff",
        "CONTRIBUTING.md",
        "docs/benchmark_card.md",
        "docs/reproducibility_card.md",
        "docs/custom_model_adapter.md",
        "docs/public_claims.md",
    ]:
        assert Path(path).exists()
        assert Path(path).stat().st_size > 0
    assert validate_readme() == []


def test_public_release_quickstart_commands_are_visible() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    repro = Path("docs/reproducibility_card.md").read_text(encoding="utf-8")
    for command in QUICKSTART_COMMANDS:
        assert command in readme
        assert command in repro


def test_public_release_claim_scanner_flags_positive_claim(tmp_path: Path) -> None:
    doc = tmp_path / "bad.md"
    doc.write_text("This is a robust calibrated refusal method with general simulator reliability.\n", encoding="utf-8")
    hits = scan_claim_language([doc])
    assert {hit["pattern"] for hit in hits} >= {"robust calibrated refusal", "general simulator reliability"}


def test_public_release_large_files_are_reviewed_or_absent() -> None:
    large_files = scan_large_files()
    assert all(row["status"] == "justified" for row in large_files)


def test_public_release_ready_script_outputs(tmp_path: Path) -> None:
    result = check_public_release_ready(tmp_path / "audit")
    assert result["verdict"] == "PUBLIC_RELEASE_READY"
    assert (tmp_path / "audit" / "public_release_check.json").exists()
    assert (tmp_path / "audit" / "private_pattern_hits.csv").exists()
    assert (tmp_path / "audit" / "large_files.csv").exists()
    assert (tmp_path / "audit" / "claim_language_hits.csv").exists()
