from __future__ import annotations

import hashlib
from pathlib import Path

from scs.experiments.v2 import validate_event_targets
from v2_fixtures import write_tiny_v2_config


def test_v2_scripts_do_not_mutate_protocol_lock(tmp_path: Path) -> None:
    protocol = Path("docs/v2/v2_scientific_protocol_lock.md")
    before = hashlib.sha256(protocol.read_bytes()).hexdigest()
    config = write_tiny_v2_config(tmp_path / "tiny.yaml", systems=["cstr", "heat_exchanger"])
    validate_event_targets(config, "configs/v2/v2_event_targets.yaml", tmp_path / "events")
    after = hashlib.sha256(protocol.read_bytes()).hexdigest()
    assert after == before
