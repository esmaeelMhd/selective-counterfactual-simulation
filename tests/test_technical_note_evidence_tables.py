from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from scs.experiments.technical_note_package import build_technical_note_evidence_tables, extract_evidence_tables


def _write_fixture_package(tmp_path: Path) -> Path:
    (tmp_path / "current_status_decision_gate.json").write_text(json.dumps({"decision": "CURRENT_STATUS_SYNCED", "allowed_next_action": "MAINTAIN_REPO_AS_WEAK_POSITIVE_BENCHMARK", "expansion_allowed": False}), encoding="utf-8")
    (tmp_path / "practical_utility_decision_gate.json").write_text(json.dumps({"decision": "NARROW_TO_WEAK_LOW_COVERAGE_CLAIM", "effect_size_verdict": "WEAK_TWO_SYSTEM_EFFECT", "expansion_allowed": False}), encoding="utf-8")
    (tmp_path / "repair_signal_role_decision_gate.json").write_text(json.dumps({"decision": "MARK_REPAIR_DIAGNOSTIC_ONLY_FOR_CSTR", "allowed_next_action": "UPDATE_SIGNAL_SEMANTICS_ONLY", "expansion_allowed": False}), encoding="utf-8")
    for name in ["current_status_decision_gate.md", "practical_utility_decision_gate.md", "repair_signal_role_decision_gate.md"]:
        (tmp_path / name).write_text("# gate\n", encoding="utf-8")
    manifest = {
        "status_id": "current_evidence_status_v1",
        "systems": {
            "two_tank": {"effect_strength": "practically_meaningful"},
            "cstr": {"effect_strength": "positive_but_weak"},
        },
        "signal_roles": {
            "repair_amount": {"cstr_role": "diagnostic_only", "cstr_repair_auroc": 0.5},
            "invariant_residual": {"cstr_role": "informative_refusal_signal", "cstr_invariant_auroc": 0.91},
        },
        "effect_size_verdict": "WEAK_TWO_SYSTEM_EFFECT",
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    pd.DataFrame(
        [
            {"coverage": 0.05, "baseline_far": 0.7, "calibrated_far": 0.4, "margin": 0.3},
            {"coverage": 0.10, "baseline_far": 0.6, "calibrated_far": 0.5, "margin": 0.1},
        ]
    ).to_csv(tmp_path / "twotank.csv", index=False)
    pd.DataFrame(
        [
            {"coverage": 0.05, "baseline_far": 0.66, "calibrated_far": 0.62, "margin": 0.04},
            {"coverage": 0.10, "baseline_far": 0.65, "calibrated_far": 0.61, "margin": 0.04},
        ]
    ).to_csv(tmp_path / "cstr.csv", index=False)
    pd.DataFrame(
        [
            {"model_id": "hold_last", "split": "id_test", "rmse_mean": 0.5},
            {"model_id": "linear_narx", "split": "id_test", "rmse_mean": 0.05},
            {"model_id": "mlp_state_space", "split": "id_test", "rmse_mean": 0.01},
        ]
    ).to_csv(tmp_path / "models.csv", index=False)
    (tmp_path / "effect.json").write_text("{}", encoding="utf-8")
    (tmp_path / "cstr_weakness.md").write_text("CSTR positive but weak", encoding="utf-8")
    config = yaml.safe_load(Path("configs/status/technical_note_package.yaml").read_text(encoding="utf-8"))
    config["controlling_status"] = {
        "current_status_gate": str(tmp_path / "current_status_decision_gate.md"),
        "practical_utility_gate": str(tmp_path / "practical_utility_decision_gate.md"),
        "repair_signal_role_gate": str(tmp_path / "repair_signal_role_decision_gate.md"),
    }
    config["source_artifacts"] = {
        "current_manifest": str(tmp_path / "manifest.json"),
        "twotank_low_coverage": str(tmp_path / "twotank.csv"),
        "cstr_low_coverage": str(tmp_path / "cstr.csv"),
        "effect_size_summary": str(tmp_path / "effect.json"),
        "cstr_weakness_diagnosis": str(tmp_path / "cstr_weakness.md"),
        "repair_signal_role_decision": str(tmp_path / "repair_signal_role_decision_gate.md"),
        "signal_semantics_registry": str(tmp_path / "manifest.json"),
        "smoke_model_metrics": str(tmp_path / "models.csv"),
    }
    path = tmp_path / "package.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_evidence_tables_extract_fixture_values(tmp_path: Path) -> None:
    config = _write_fixture_package(tmp_path)
    tables = extract_evidence_tables(config)

    main = tables["main"]
    cstr = main[(main["system"] == "cstr") & (main["coverage"] == 0.05)].iloc[0]
    assert cstr["margin"] == 0.04
    assert cstr["effect_strength"] == "positive_but_weak"
    assert tables["summary"]["repair_signal_role_decision"] == "MARK_REPAIR_DIAGNOSTIC_ONLY_FOR_CSTR"
    assert set(tables["models"]["model"]) == {"hold_last", "linear_narx", "mlp_state_space"}


def test_evidence_table_report_contains_real_values(tmp_path: Path) -> None:
    config = _write_fixture_package(tmp_path)
    build_technical_note_evidence_tables(config, tmp_path / "tables", report_output=tmp_path / "report.md")
    text = (tmp_path / "report.md").read_text(encoding="utf-8")

    assert "0.300000" in text
    assert "positive_but_weak" in text
    assert "diagnostic_only" in text
