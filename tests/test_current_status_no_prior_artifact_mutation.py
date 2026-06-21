from __future__ import annotations

import hashlib
import json
from pathlib import Path


def test_current_status_preconditions_record_no_prior_artifact_mutation() -> None:
    manifest = json.loads(Path("results/current_status/preconditions/prior_artifact_hashes.json").read_text(encoding="utf-8"))

    assert manifest["artifacts"]
    for path, expected in manifest["artifacts"].items():
        current = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        assert current == expected["sha256"], path
