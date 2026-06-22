from __future__ import annotations

import hashlib
import json
from pathlib import Path


def test_technical_note_source_artifacts_not_mutated() -> None:
    hashes = json.loads(Path("results/technical_note_package/preconditions/source_artifact_hashes.json").read_text(encoding="utf-8"))

    assert hashes["artifacts"]
    for _, item in hashes["artifacts"].items():
        assert hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() == item["sha256"]
