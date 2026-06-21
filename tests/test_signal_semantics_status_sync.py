from __future__ import annotations

import json
from pathlib import Path

from scs.validators.signal_semantics import signal_semantics_registry


def test_signal_semantics_registry_matches_current_manifest() -> None:
    manifest = json.loads(Path("results/current_status/evidence_manifest/current_evidence_manifest.json").read_text(encoding="utf-8"))
    registry = signal_semantics_registry()

    assert registry["repair_amount"]["cstr_role"] == manifest["signal_roles"]["repair_amount"]["cstr_role"]
    assert registry["repair_amount"]["twotank_role"] == "diagnostic_constraint_signal"
    assert registry["repair_amount"]["universal_refusal_signal"] is False
    assert registry["invariant_residual"]["cstr_role"] == manifest["signal_roles"]["invariant_residual"]["cstr_role"]
    assert registry["invariant_residual"]["universal_refusal_signal"] is False


def test_signal_semantics_sync_artifacts_include_roles() -> None:
    docs_text = Path("docs/signal_semantics_registry.md").read_text(encoding="utf-8")
    report_text = Path("reports/signal_semantics_status_sync.md").read_text(encoding="utf-8")

    assert "diagnostic_only" in docs_text
    assert "informative_refusal_signal" in docs_text
    assert "SIGNAL_SEMANTICS_SYNCED" in report_text
