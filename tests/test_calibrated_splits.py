from __future__ import annotations

from pathlib import Path

from conftest import write_tiny_calibrated_config
from scs.experiments.calibrated import generate_calibrated_data, load_calibrated_config


def test_calibrated_splits_are_distinct_and_shifted(tmp_path: Path) -> None:
    config_path = write_tiny_calibrated_config(tmp_path / "tiny.yaml")
    config = load_calibrated_config(config_path)
    dataset = generate_calibrated_data(config, tmp_path / "out")

    calibration_ids = {
        f"{split}_{i:04d}"
        for split, batch in dataset.items()
        if split.startswith("judge_calibration")
        for i in range(batch.n_trajectories)
    }
    test_ids = {
        f"{split}_{i:04d}"
        for split, batch in dataset.items()
        if split.startswith("judge_test")
        for i in range(batch.n_trajectories)
    }
    assert calibration_ids.isdisjoint(test_ids)
    assert "model_train" in dataset

    split_summary = (tmp_path / "out" / "split_summary.csv").read_text(encoding="utf-8")
    assert "judge_calibration_ood_action_magnitude" in split_summary
    assert "judge_test_ood_combined" in split_summary

    integrity = (tmp_path / "out" / "split_integrity.json").read_text(encoding="utf-8")
    assert "scenario_overlap_count" in integrity
    assert "trajectory_overlap_count" in integrity
