from __future__ import annotations

from pathlib import Path

import pytest

import scs.experiments.cstr_weakness as cwa


def test_practical_gate_decision_is_read_and_expansion_is_blocked(tmp_path: Path) -> None:
    result = cwa.verify_cstr_weakness_audit_preconditions(
        "configs/audits/cstr_weakness_audit.yaml",
        tmp_path / "preconditions",
        report_output=None,
    )
    assert result["current_controlling_decision"] == "NARROW_TO_WEAK_LOW_COVERAGE_CLAIM"
    assert result["expansion_allowed"] is False
    assert result["verdict"] == "READY_FOR_CSTR_WEAKNESS_AUDIT"


def test_missing_artifact_causes_failure(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cwa, "PRIOR_ARTIFACTS", ["results/does_not_exist_for_cstr_weakness.csv"])
    with pytest.raises(RuntimeError):
        cwa.verify_cstr_weakness_audit_preconditions(
            "configs/audits/cstr_weakness_audit.yaml",
            tmp_path / "preconditions",
            report_output=None,
        )


def test_old_repo_imports_and_path_hacks_are_detected(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    old_import = "import time" + "_series" + "_simulator"
    path_hack = "sys" + ".path" + ".append('/tmp/old')"
    (source / "bad.py").write_text(
        f"{old_import}\n{path_hack}\n",
        encoding="utf-8",
    )
    scan = cwa._scan_forbidden_runtime_refs([source])
    assert scan["old_repo_runtime_import_hits"]
    assert scan["path_hack_hits"]
