from __future__ import annotations

import json
from pathlib import Path

from scs.reports.claim_downgrade import make_claim_downgrade


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_claim_downgrade_generated_from_gate_artifacts(tmp_path) -> None:
    decision = tmp_path / "decision.json"
    freeze = tmp_path / "freeze.json"
    claim = tmp_path / "claim.json"
    seed = tmp_path / "seed.json"
    severity = tmp_path / "severity.json"
    output_md = tmp_path / "downgrade.md"
    output_json = tmp_path / "downgrade.json"

    _write_json(decision, {"decision": "KILL_OR_DOWNGRADE_CLAIM", "required_next_action": "downgrade"})
    _write_json(freeze, {"verdict": "ACCEPTED"})
    _write_json(claim, {"verdict": "NOT_SUPPORTED", "overall_win_rate": 0.2})
    _write_json(seed, {"verdict": "NOT_SUPPORTED", "aggregate": {"overall_combined_win_rate": {"mean": 0.1}}})
    _write_json(severity, {"verdict": "MEANINGFUL"})

    report = make_claim_downgrade(decision, freeze, claim, seed, severity, output_md, output_json)

    assert report["action"] == "DOWNGRADE_CLAIM"
    assert output_md.exists()
    text = output_md.read_text(encoding="utf-8")
    assert "NOT_SUPPORTED" in text
    assert "combined_linear remains an exploratory baseline" in text
    assert json.loads(output_json.read_text(encoding="utf-8"))["action"] == "DOWNGRADE_CLAIM"

