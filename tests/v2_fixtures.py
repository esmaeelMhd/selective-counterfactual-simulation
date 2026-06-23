from __future__ import annotations

from pathlib import Path

import yaml


def write_tiny_v2_config(path: Path, systems: list[str] | None = None, models: list[str] | None = None) -> Path:
    config = {
        "test_mode": True,
        "v2_id": "v2_scientific_strengthening",
        "source_v1_tag": "v1.1-benchmark-usability",
        "systems": systems or ["two_tank", "cstr"],
        "models": models or ["hold_last", "linear_narx"],
        "signals": [
            "support_distance",
            "uncertainty_score",
            "disagreement_score",
            "invariant_residual",
            "repair_amount",
        ],
        "judges": {
            "simple": [
                "support_only",
                "uncertainty_only",
                "disagreement_only",
                "invariant_only",
                "repair_only",
                "random_baseline",
            ],
            "calibrated": [
                "best_single_signal_selected_on_calibration",
                "calibration_selected_candidate_ranker",
                "rank_normalized_linear",
                "logistic_calibrated_judge",
                "isotonic_calibrated_judge",
                "quantile_rule_judge",
                "conservative_low_coverage_judge",
            ],
            "stronger_baselines": [
                "learned_error_classifier",
                "conformal_risk_threshold",
                "ensemble_disagreement_threshold",
            ],
            "diagnostic_only": ["oracle_error_rank"],
        },
        "primary_calibrated_judge": "calibration_selected_candidate_ranker",
        "primary_coverages": [0.05, 0.10],
        "coverage_grid": [0.01, 0.02, 0.05, 0.10, 0.20],
        "badness_targets": ["bad_rmse", "bad_event", "bad_rmse_or_event"],
        "rmse_thresholds": [0.05, 0.10, 0.15, 0.20, 0.30, 0.50],
        "seeds": [0, 1],
        "practical_thresholds": {
            "minimum_absolute_far_reduction": 0.05,
            "minimum_relative_far_reduction": 0.10,
            "minimum_seed_win_rate_strong": 0.70,
        },
        "data": {
            "horizon": 8,
            "dt": 0.1,
            "n_model_train": 8,
            "n_calibration_id": 3,
            "n_calibration_ood": 3,
            "n_test_id": 4,
            "n_test_ood": 4,
            "uncertainty_samples": 1,
        },
        "model_runtime": {
            "mlp_max_iter": 12,
            "ensemble_members": 2,
            "ensemble_mlp_max_iter": 10,
            "gradient_boosting_max_iter": 12,
        },
        "heat_exchanger_event_thresholds": {
            "outlet_temperature_high": 101.0,
            "outlet_temperature_low": 76.0,
            "tracking_error_high": 10.0,
            "target_outlet_temperature": 88.0,
        },
        "source_artifacts": ["README.md"],
        "forbidden": {
            "allow_product_features": False,
            "allow_protocol_mutation_after_results": False,
            "allow_test_label_selection": False,
            "allow_oracle_as_real_method": False,
            "allow_v1_artifact_overwrite": False,
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path
