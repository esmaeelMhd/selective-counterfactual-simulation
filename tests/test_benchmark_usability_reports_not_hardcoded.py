from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from scs.experiments.benchmark_usability import build_benchmark_card, write_demo_report


def test_benchmark_card_changes_with_manifest_fixture(tmp_path: Path) -> None:
    config = yaml.safe_load(Path("configs/status/benchmark_usability_v1_1.yaml").read_text(encoding="utf-8"))
    manifest = {
        "systems": {
            "two_tank": {"coverage_0_05_margin": 0.321},
            "cstr": {"coverage_0_05_margin": 0.012},
        }
    }
    config_path = tmp_path / "config.yaml"
    manifest_path = tmp_path / "manifest.json"
    output = tmp_path / "card.md"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    manifest_path.write_text(__import__("json").dumps(manifest), encoding="utf-8")

    build_benchmark_card(config_path, manifest_path, output)

    text = output.read_text(encoding="utf-8")
    assert "0.321000" in text
    assert "0.012000" in text


def test_demo_report_changes_with_result_table_fixture(tmp_path: Path) -> None:
    table = pd.DataFrame(
        [
            {
                "system_id": "two_tank",
                "coverage": 0.05,
                "baseline_judge": "fixture_baseline",
                "calibrated_judge": "fixture_demo",
                "baseline_far": 0.9,
                "calibrated_far": 0.4,
                "absolute_margin": 0.5,
                "claim_scope": "fixture_scope",
                "is_demo": True,
            }
        ]
    )
    output = tmp_path / "demo.md"
    write_demo_report(table, {"allowed_claim": "fixture weak claim"}, output)

    text = output.read_text(encoding="utf-8")
    assert "fixture_demo" in text
    assert "0.500000" in text
    assert "fixture weak claim" in text
