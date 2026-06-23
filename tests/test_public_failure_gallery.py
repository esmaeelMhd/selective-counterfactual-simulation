from __future__ import annotations

import json
from pathlib import Path


def test_public_failure_gallery_uses_real_artifact_rows() -> None:
    manifest = json.loads(Path("results/v2_public_benchmark_hardening/failure_gallery_manifest.json").read_text())
    assert manifest["verdict"] == "PUBLIC_FAILURE_GALLERY_BUILT"
    assert manifest["example_count"] >= 5
    examples = manifest["examples"]
    assert any(example["false_accept"] and example["event_bad"] for example in examples)
    assert any((not example["accepted"]) and (not example["false_accept"]) for example in examples)
    assert all(isinstance(example["source_row_id"], int) for example in examples)
    text = Path("docs/v2/event_risk_failure_gallery.md").read_text(encoding="utf-8")
    assert "Trajectory plot unavailable from current artifacts." in text
    assert "fabricated" not in text.lower()
