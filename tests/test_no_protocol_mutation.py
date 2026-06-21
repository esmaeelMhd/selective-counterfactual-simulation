from __future__ import annotations

from pathlib import Path

import yaml

from conftest import write_tiny_calibrated_cstr_config
from scs.experiments.cstr_replication import (
    FROZEN_CANDIDATE_JUDGES,
    FROZEN_PRIMARY_COVERAGES,
    run_cstr_sanity,
    validate_protocol_lock,
)


def test_cstr_config_does_not_mutate_frozen_protocol() -> None:
    protocol = validate_protocol_lock("docs/calibrated_protocol_lock_v1.md")
    config = yaml.safe_load(Path("configs/experiments/calibrated_cstr.yaml").read_text(encoding="utf-8"))
    configured_calibrated = [judge for judge in config["judges"] if judge in FROZEN_CANDIDATE_JUDGES]
    assert configured_calibrated == FROZEN_CANDIDATE_JUDGES
    assert [float(value) for value in config["primary_coverages"]] == FROZEN_PRIMARY_COVERAGES
    assert protocol["valid"] is True


def test_cstr_scripts_do_not_modify_protocol_lock(tmp_path: Path) -> None:
    protocol_path = Path("docs/calibrated_protocol_lock_v1.md")
    before = protocol_path.read_text(encoding="utf-8")
    config = write_tiny_calibrated_cstr_config(tmp_path / "tiny_cstr.yaml")
    run_cstr_sanity(config, tmp_path / "sanity")
    after = protocol_path.read_text(encoding="utf-8")
    assert before == after
