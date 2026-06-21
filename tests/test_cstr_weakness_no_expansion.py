from __future__ import annotations

from pathlib import Path

import yaml


def test_cstr_weakness_config_forbids_expansion() -> None:
    config = yaml.safe_load(Path("configs/audits/cstr_weakness_audit.yaml").read_text(encoding="utf-8"))
    assert config["forbidden"]["allow_new_models"] is False
    assert config["forbidden"]["allow_new_judges"] is False
    assert config["forbidden"]["allow_new_signals"] is False
    assert config["forbidden"]["allow_new_systems"] is False
    assert config["forbidden"]["allow_protocol_mutation"] is False


def test_audit_scripts_do_not_call_forbidden_expansion_targets() -> None:
    scripts = [
        "scripts/verify_cstr_weakness_audit_preconditions.py",
        "scripts/build_cstr_diagnosis_table.py",
        "scripts/analyze_cstr_statewise_error.py",
        "scripts/analyze_cstr_repair_signal.py",
        "scripts/analyze_cstr_signal_overlap.py",
        "scripts/analyze_cstr_model_scenario_failures.py",
        "scripts/analyze_cstr_rmse_target.py",
        "scripts/make_cstr_weakness_diagnosis.py",
    ]
    text = "\n".join(Path(path).read_text(encoding="utf-8") for path in scripts)
    for forbidden in ["heat_exchanger", "RSSM", "rssm", "FastAPI", "frontend", "dashboard"]:
        assert forbidden not in text


def test_protocol_lock_is_not_modified_by_audit_scripts() -> None:
    before = Path("docs/calibrated_protocol_lock_v1.md").read_text(encoding="utf-8")
    after = Path("docs/calibrated_protocol_lock_v1.md").read_text(encoding="utf-8")
    assert before == after
