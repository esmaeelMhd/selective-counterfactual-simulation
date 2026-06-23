from __future__ import annotations

import yaml

from scs.experiments.v2 import generate_v2_dataset, load_v2_config, split_overlap_report


def test_v2_detects_calibration_test_overlap(tmp_path) -> None:
    config = load_v2_config("configs/v2/v2_scientific_strengthening.yaml")
    dataset = generate_v2_dataset("two_tank", config, seed=0)
    cal = dataset["judge_calibration_id"]
    test = dataset["judge_test_id"]
    cal.states[0] = test.states[0]
    cal.actions[0] = test.actions[0]
    cal.disturbances[0] = test.disturbances[0]
    overlap = split_overlap_report(dataset)
    assert overlap["overlap_count"] > 0


def test_v2_oracle_is_diagnostic_only() -> None:
    config = yaml.safe_load(open("configs/v2/v2_scientific_strengthening.yaml", encoding="utf-8"))
    assert "oracle_error_rank" in config["judges"]["diagnostic_only"]
    assert "oracle_error_rank" not in config["judges"]["simple"]
    assert "oracle_error_rank" not in config["judges"]["calibrated"]
    assert config["forbidden"]["allow_test_label_selection"] is False
    assert config["forbidden"]["allow_oracle_as_real_method"] is False
