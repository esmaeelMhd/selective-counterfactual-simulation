from __future__ import annotations

import json
from pathlib import Path


def test_source_artifact_hashes_unchanged_after_public_scripts() -> None:
    hashes = json.loads(Path("results/v2_public_benchmark_hardening/preconditions/source_artifact_hashes.json").read_text())
    for row in hashes.values():
        path = Path(row["path"])
        assert path.exists()
        import hashlib

        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
