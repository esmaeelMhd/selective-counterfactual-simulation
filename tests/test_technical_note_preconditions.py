from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scs.experiments.technical_note_package import load_package_config, verify_technical_note_preconditions


def test_technical_note_preconditions_write_source_hashes(tmp_path: Path) -> None:
    result = verify_technical_note_preconditions(
        "configs/status/technical_note_package.yaml",
        tmp_path / "preconditions",
        report_output=tmp_path / "report.md",
    )

    hashes = json.loads((tmp_path / "preconditions" / "source_artifact_hashes.json").read_text(encoding="utf-8"))
    assert result["verdict"] == "READY_FOR_TECHNICAL_NOTE_PACKAGE"
    assert hashes["artifacts"]
    assert all(item["sha256"] for item in hashes["artifacts"].values())


def test_missing_current_status_gate_fails(tmp_path: Path) -> None:
    config = yaml.safe_load(Path("configs/status/technical_note_package.yaml").read_text(encoding="utf-8"))
    config["controlling_status"]["current_status_gate"] = str(tmp_path / "missing_current_gate.md")
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        verify_technical_note_preconditions(path, tmp_path / "out", report_output=tmp_path / "report.md")


def test_missing_practical_gate_fails(tmp_path: Path) -> None:
    config = yaml.safe_load(Path("configs/status/technical_note_package.yaml").read_text(encoding="utf-8"))
    config["controlling_status"]["practical_utility_gate"] = str(tmp_path / "missing_practical_gate.md")
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        verify_technical_note_preconditions(path, tmp_path / "out", report_output=tmp_path / "report.md")


def test_expansion_allowed_true_fails(tmp_path: Path) -> None:
    config = yaml.safe_load(Path("configs/status/technical_note_package.yaml").read_text(encoding="utf-8"))
    config["forbidden"]["allow_new_experiments"] = True
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="allow_new_experiments"):
        load_package_config(path)
