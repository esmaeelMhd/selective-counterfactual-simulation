from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scs.experiments.current_status import load_current_status_config


def test_current_status_config_preserves_weak_claim_and_no_expansion() -> None:
    config = load_current_status_config("configs/status/current_evidence_status.yaml")

    assert config["status_id"] == "current_evidence_status_v1"
    assert config["current_allowed_claim"]["label"] == "WEAK_LOW_COVERAGE_CLAIM"
    assert config["expansion_allowed"] is False
    assert config["signal_role_decisions"]["repair_amount"]["cstr_role"] == "diagnostic_only"
    assert config["signal_role_decisions"]["repair_amount"]["universal_refusal_signal"] is False
    assert config["signal_role_decisions"]["invariant_residual"]["cstr_role"] == "informative_refusal_signal"


def test_current_status_config_rejects_expansion_allowed(tmp_path: Path) -> None:
    config = yaml.safe_load(Path("configs/status/current_evidence_status.yaml").read_text(encoding="utf-8"))
    config["expansion_allowed"] = True
    path = tmp_path / "bad_status.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="expansion_allowed"):
        load_current_status_config(path)
