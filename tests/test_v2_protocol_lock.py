from __future__ import annotations

from pathlib import Path

import yaml


def test_v2_protocol_lock_matches_config() -> None:
    config = yaml.safe_load(Path("configs/v2/v2_scientific_strengthening.yaml").read_text(encoding="utf-8"))
    text = Path("docs/v2/v2_scientific_protocol_lock.md").read_text(encoding="utf-8")

    required_sections = [
        "Starting v1 status",
        "v2 research question",
        "Systems",
        "Model participants",
        "Refusal signals",
        "Judges and baselines",
        "Calibration/test separation",
        "Coverage grid",
        "Threshold grid",
        "Forbidden changes after results",
    ]
    for section in required_sections:
        assert f"## {section}" in text
    for value in config["systems"] + config["models"] + config["signals"]:
        assert f"- {value}" in text
    for coverage in config["coverage_grid"]:
        assert f"- {coverage:.2f}" in text
    for threshold in config["rmse_thresholds"]:
        assert f"- {threshold:.2f}" in text
    assert "Do not add a new judge after a failure" in text
    assert config["forbidden"]["allow_protocol_mutation_after_results"] is False
