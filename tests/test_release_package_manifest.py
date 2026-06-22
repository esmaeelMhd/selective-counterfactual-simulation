from __future__ import annotations

import json
from pathlib import Path


def test_release_package_manifest_paths_and_hashes_exist() -> None:
    manifest = json.loads(Path("results/technical_note_package/package_manifest.json").read_text(encoding="utf-8"))

    assert manifest["source_artifact_hashes"]
    assert manifest["known_limitations"]
    assert manifest["forbidden_claims"]
    for key in ["included_docs", "included_reports", "included_figures"]:
        assert manifest[key]
        for path in manifest[key]:
            assert Path(path).exists(), path
            assert Path(path).stat().st_size > 0, path


def test_release_note_blocks_expansion() -> None:
    text = Path("reports/release_note_v1_current_status.md").read_text(encoding="utf-8")
    assert "Expansion is blocked" in text
    assert "weak-positive benchmark" in text
