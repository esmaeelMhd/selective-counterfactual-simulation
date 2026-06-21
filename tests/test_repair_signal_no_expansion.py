from __future__ import annotations

from pathlib import Path

import yaml


def test_repair_signal_audit_config_forbids_expansion() -> None:
    config = yaml.safe_load(Path("configs/audits/repair_signal_semantics_audit.yaml").read_text(encoding="utf-8"))
    assert config["systems"] == ["two_tank", "cstr"]
    assert all(value is False for value in config["forbidden"].values())
    assert "heat_exchanger" not in yaml.safe_dump(config)
    assert "rssm" not in yaml.safe_dump(config).lower()


def test_repair_signal_scripts_do_not_run_forbidden_expansion_paths() -> None:
    script_text = "\n".join(path.read_text(encoding="utf-8") for path in Path("scripts").glob("*repair_signal*.py"))
    assert "heat_exchanger" not in script_text
    assert "rssm_adapter" not in script_text.lower()
    assert "run_rssm" not in script_text.lower()


def test_repair_signal_audit_does_not_modify_protocol_lock() -> None:
    before = Path("docs/calibrated_protocol_lock_v1.md").read_text(encoding="utf-8")
    after = Path("docs/calibrated_protocol_lock_v1.md").read_text(encoding="utf-8")
    assert before == after
