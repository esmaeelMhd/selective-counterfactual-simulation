from __future__ import annotations

import json
from pathlib import Path

from scs.validators.signal_semantics import REQUIRED_SIGNAL_IDS, signal_semantics_registry, write_signal_semantics_artifacts


def test_signal_semantics_registry_has_required_fields_and_repair_blind_spots() -> None:
    registry = signal_semantics_registry()
    required_fields = {
        "signal_id",
        "description",
        "risk_orientation",
        "expected_higher_means",
        "requires_trajectory",
        "requires_bounds",
        "requires_repair_operator",
        "requires_ensemble",
        "system_applicability",
        "failure_type_detected",
        "failure_type_not_detected",
        "known_blind_spots",
        "is_universal_candidate",
        "is_system_specific_candidate",
    }
    assert set(registry) == set(REQUIRED_SIGNAL_IDS)
    for signal in REQUIRED_SIGNAL_IDS:
        assert required_fields <= set(registry[signal])
        assert registry[signal]["is_universal_candidate"] is False
    repair = registry["repair_amount"]
    blind_text = " ".join(repair["known_blind_spots"] + repair["failure_type_not_detected"])
    assert "within-bound dynamic errors" in blind_text or "within-bound dynamical error" in blind_text
    assert repair["requires_bounds"] is True
    assert repair["requires_repair_operator"] is True


def test_signal_semantics_report_is_generated_from_registry(tmp_path: Path) -> None:
    artifacts = write_signal_semantics_artifacts(
        tmp_path / "report.md",
        results_output=tmp_path / "results",
        docs_output=tmp_path / "docs.md",
    )
    registry = json.loads(Path(artifacts["registry_json"]).read_text(encoding="utf-8"))
    report = Path(artifacts["report"]).read_text(encoding="utf-8")
    assert registry["repair_amount"]["description"] in report
    assert "No signal is treated as universally reliable." in report
