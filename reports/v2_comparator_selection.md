# v2 Comparator Selection

## Verdict

COMPARATOR_SELECTION_VALID

## Selection Rule

All deployable baselines are selected on calibration rows only. The primary metric is lowest false accept rate at coverage 0.05 and 0.10. Tie-breakers are lower accepted error, higher achieved coverage, then alphabetical `judge_id`.

## Global Baseline Selection Preview

| selection_mode | seed | coverage | selected_judge_id | selection_far | source_split | uses_test_labels |
| --- | ---: | ---: | --- | ---: | --- | ---: |
| global_calibration_selected_baseline | 0 | 0.050000 | learned_error_classifier | 0.000000 | calibration | False |
| global_calibration_selected_baseline | 0 | 0.100000 | learned_error_classifier | 0.000000 | calibration | False |
| global_calibration_selected_baseline | 1 | 0.050000 | learned_error_classifier | 0.000000 | calibration | False |
| global_calibration_selected_baseline | 1 | 0.100000 | learned_error_classifier | 0.003036 | calibration | False |
| global_calibration_selected_baseline | 2 | 0.050000 | learned_error_classifier | 0.000000 | calibration | False |
| global_calibration_selected_baseline | 2 | 0.100000 | learned_error_classifier | 0.014170 | calibration | False |
| global_calibration_selected_baseline | 3 | 0.050000 | learned_error_classifier | 0.004049 | calibration | False |
| global_calibration_selected_baseline | 3 | 0.100000 | learned_error_classifier | 0.010121 | calibration | False |
| global_calibration_selected_baseline | 4 | 0.050000 | conformal_risk_threshold | 0.000000 | calibration | False |
| global_calibration_selected_baseline | 4 | 0.100000 | conformal_risk_threshold | 0.007085 | calibration | False |
| global_calibration_selected_baseline | 5 | 0.050000 | learned_error_classifier | 0.004049 | calibration | False |
| global_calibration_selected_baseline | 5 | 0.100000 | learned_error_classifier | 0.005061 | calibration | False |

## Per-System-Target Selection Preview

| selection_mode | seed | system_id | badness_target | coverage | selected_judge_id | selection_far |
| --- | ---: | --- | --- | ---: | --- | ---: |
| per_system_target_calibration_selected_baseline | 0 | cstr | bad_event | 0.050000 | conformal_risk_threshold | 0.000000 |
| per_system_target_calibration_selected_baseline | 0 | cstr | bad_event | 0.100000 | conformal_risk_threshold | 0.000000 |
| per_system_target_calibration_selected_baseline | 0 | cstr | bad_rmse | 0.050000 | conformal_risk_threshold | 0.000000 |
| per_system_target_calibration_selected_baseline | 0 | cstr | bad_rmse | 0.100000 | conformal_risk_threshold | 0.000000 |
| per_system_target_calibration_selected_baseline | 0 | cstr | bad_rmse_or_event | 0.050000 | conformal_risk_threshold | 0.000000 |
| per_system_target_calibration_selected_baseline | 0 | cstr | bad_rmse_or_event | 0.100000 | conformal_risk_threshold | 0.000000 |
| per_system_target_calibration_selected_baseline | 0 | heat_exchanger | bad_event | 0.050000 | conformal_risk_threshold | 0.000000 |
| per_system_target_calibration_selected_baseline | 0 | heat_exchanger | bad_event | 0.100000 | conformal_risk_threshold | 0.000000 |
| per_system_target_calibration_selected_baseline | 0 | heat_exchanger | bad_rmse | 0.050000 | conformal_risk_threshold | 0.000000 |
| per_system_target_calibration_selected_baseline | 0 | heat_exchanger | bad_rmse | 0.100000 | conformal_risk_threshold | 0.000000 |
| per_system_target_calibration_selected_baseline | 0 | heat_exchanger | bad_rmse_or_event | 0.050000 | conformal_risk_threshold | 0.000000 |
| per_system_target_calibration_selected_baseline | 0 | heat_exchanger | bad_rmse_or_event | 0.100000 | conformal_risk_threshold | 0.000000 |
| per_system_target_calibration_selected_baseline | 0 | two_tank | bad_event | 0.050000 | conformal_risk_threshold | 0.000000 |
| per_system_target_calibration_selected_baseline | 0 | two_tank | bad_event | 0.100000 | conformal_risk_threshold | 0.000000 |
| per_system_target_calibration_selected_baseline | 0 | two_tank | bad_rmse | 0.050000 | learned_error_classifier | 0.000000 |
| per_system_target_calibration_selected_baseline | 0 | two_tank | bad_rmse | 0.100000 | learned_error_classifier | 0.000000 |
| per_system_target_calibration_selected_baseline | 0 | two_tank | bad_rmse_or_event | 0.050000 | learned_error_classifier | 0.000000 |
| per_system_target_calibration_selected_baseline | 0 | two_tank | bad_rmse_or_event | 0.100000 | learned_error_classifier | 0.000000 |

## Calibrated-Family Selection Preview

| selection_mode | seed | system_id | badness_target | coverage | selected_judge_id | selection_far |
| --- | ---: | --- | --- | ---: | --- | ---: |
| best_calibrated_family_selected_on_calibration | 0 | cstr | bad_event | 0.050000 | best_single_signal_selected_on_calibration | 0.000000 |
| best_calibrated_family_selected_on_calibration | 0 | cstr | bad_event | 0.100000 | best_single_signal_selected_on_calibration | 0.000000 |
| best_calibrated_family_selected_on_calibration | 0 | cstr | bad_rmse | 0.050000 | best_single_signal_selected_on_calibration | 0.000000 |
| best_calibrated_family_selected_on_calibration | 0 | cstr | bad_rmse | 0.100000 | best_single_signal_selected_on_calibration | 0.000000 |
| best_calibrated_family_selected_on_calibration | 0 | cstr | bad_rmse_or_event | 0.050000 | best_single_signal_selected_on_calibration | 0.000000 |
| best_calibrated_family_selected_on_calibration | 0 | cstr | bad_rmse_or_event | 0.100000 | best_single_signal_selected_on_calibration | 0.000000 |
| best_calibrated_family_selected_on_calibration | 0 | heat_exchanger | bad_event | 0.050000 | best_single_signal_selected_on_calibration | 0.000000 |
| best_calibrated_family_selected_on_calibration | 0 | heat_exchanger | bad_event | 0.100000 | best_single_signal_selected_on_calibration | 0.000000 |
| best_calibrated_family_selected_on_calibration | 0 | heat_exchanger | bad_rmse | 0.050000 | best_single_signal_selected_on_calibration | 0.000000 |
| best_calibrated_family_selected_on_calibration | 0 | heat_exchanger | bad_rmse | 0.100000 | best_single_signal_selected_on_calibration | 0.000000 |
| best_calibrated_family_selected_on_calibration | 0 | heat_exchanger | bad_rmse_or_event | 0.050000 | best_single_signal_selected_on_calibration | 0.000000 |
| best_calibrated_family_selected_on_calibration | 0 | heat_exchanger | bad_rmse_or_event | 0.100000 | best_single_signal_selected_on_calibration | 0.000000 |
| best_calibrated_family_selected_on_calibration | 0 | two_tank | bad_event | 0.050000 | best_single_signal_selected_on_calibration | 0.000000 |
| best_calibrated_family_selected_on_calibration | 0 | two_tank | bad_event | 0.100000 | best_single_signal_selected_on_calibration | 0.000000 |
| best_calibrated_family_selected_on_calibration | 0 | two_tank | bad_rmse | 0.050000 | conservative_low_coverage_judge | 0.000000 |
| best_calibrated_family_selected_on_calibration | 0 | two_tank | bad_rmse | 0.100000 | logistic_calibrated_judge | 0.000000 |
| best_calibrated_family_selected_on_calibration | 0 | two_tank | bad_rmse_or_event | 0.050000 | calibration_selected_candidate_ranker | 0.000000 |
| best_calibrated_family_selected_on_calibration | 0 | two_tank | bad_rmse_or_event | 0.100000 | logistic_calibrated_judge | 0.000000 |

## Reasons

['none']
