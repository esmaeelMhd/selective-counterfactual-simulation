from __future__ import annotations

import json
from pathlib import Path

from scs.experiments.current_status import README_END, README_START, render_readme_status_block


def test_readme_current_status_block_is_synced() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    summary = json.loads(Path("results/current_status/readme_sync/readme_status_sync_summary.json").read_text(encoding="utf-8"))

    assert summary["verdict"] == "README_SYNCED"
    assert README_START in readme
    assert README_END in readme
    assert "Expansion is currently blocked" in readme
    assert "positive but practically weak" in readme
    assert "diagnostic-only for CSTR" in readme


def test_readme_status_renderer_uses_manifest_values() -> None:
    manifest = {
        "controlling_claim_text": "fixture current claim 9.876",
        "practical_utility_decision": "FIXTURE_PRACTICAL_GATE",
        "repair_signal_role_decision": "FIXTURE_REPAIR_GATE",
    }

    block = render_readme_status_block(manifest)

    assert "fixture current claim 9.876" in block
    assert "FIXTURE_PRACTICAL_GATE" in block
    assert "FIXTURE_REPAIR_GATE" in block
