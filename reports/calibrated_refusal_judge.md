# Calibrated Refusal Judge Report

## Research question

Can a calibrated refusal judge identify a low-coverage subset of counterfactual rollouts with lower false-accept risk than the strongest calibration-selected single signal?

## Strict leakage statement

Simulator models are fit on `model_train`; calibrated judges are selected or fit on `judge_calibration_*`; final risk-coverage is computed on `judge_test_*`. Test labels are not used during judge fitting. Oracle is diagnostic only.

## Data splits

model_train, judge_calibration_id, judge_calibration_ood_action_magnitude, judge_calibration_ood_inflow_spike, judge_calibration_ood_combined, judge_calibration_pump_degradation, judge_test_id, judge_test_ood_action_magnitude, judge_test_ood_inflow_spike, judge_test_ood_combined, judge_test_pump_degradation.

## Models

hold_last, linear_narx, mlp_state_space

## Signals

support_distance, uncertainty_score, disagreement_score, invariant_residual, repair_amount

## Judges

support_only, uncertainty_only, disagreement_only, invariant_only, repair_only, combined_linear, best_single_signal_selected_on_calibration, rank_normalized_linear, logistic_calibrated_judge, isotonic_calibrated_judge, quantile_rule_judge, conservative_low_coverage_judge, random_baseline, oracle_error_rank

## Calibration provenance

| judge_id | available | selected_signal_if_any | used_test_labels_during_fit |
| --- | ---: | --- | ---: |
| best_single_signal_selected_on_calibration | True | invariant_residual | False |
| rank_normalized_linear | True | nan | False |
| logistic_calibrated_judge | True | nan | False |
| isotonic_calibrated_judge | True | invariant_residual | False |
| quantile_rule_judge | True | invariant_residual | False |
| conservative_low_coverage_judge | True | nan | False |

## Low-coverage result

| coverage | best_deployable_baseline | baseline_far | best_calibrated_judge | calibrated_far | margin |
| ---: | --- | ---: | --- | ---: | ---: |
| 0.050000 | best_single_signal_selected_on_calibration | 0.640000 | rank_normalized_linear | 0.493333 | 0.146667 |
| 0.100000 | best_single_signal_selected_on_calibration | 0.653333 | rank_normalized_linear | 0.520000 | 0.133333 |

## Full risk-coverage result

| judge_id | coverage_requested | false_accept_rate |
| --- | ---: | ---: |
| best_single_signal_selected_on_calibration | 0.050000 | 0.640000 |
| best_single_signal_selected_on_calibration | 0.100000 | 0.653333 |
| best_single_signal_selected_on_calibration | 0.200000 | 0.643333 |
| best_single_signal_selected_on_calibration | 0.400000 | 0.641667 |
| best_single_signal_selected_on_calibration | 0.600000 | 0.646667 |
| best_single_signal_selected_on_calibration | 0.800000 | 0.648333 |
| best_single_signal_selected_on_calibration | 1.000000 | 0.646667 |
| combined_linear | 0.050000 | 0.466667 |
| combined_linear | 0.100000 | 0.493333 |
| combined_linear | 0.200000 | 0.566667 |
| combined_linear | 0.400000 | 0.616667 |
| combined_linear | 0.600000 | 0.633333 |
| combined_linear | 0.800000 | 0.641667 |
| combined_linear | 1.000000 | 0.646667 |
| conservative_low_coverage_judge | 0.050000 | 0.520000 |
| conservative_low_coverage_judge | 0.100000 | 0.553333 |
| conservative_low_coverage_judge | 0.200000 | 0.580000 |
| conservative_low_coverage_judge | 0.400000 | 0.616667 |
| conservative_low_coverage_judge | 0.600000 | 0.633333 |
| conservative_low_coverage_judge | 0.800000 | 0.641667 |
| conservative_low_coverage_judge | 1.000000 | 0.646667 |
| disagreement_only | 0.050000 | 0.466667 |
| disagreement_only | 0.100000 | 0.486667 |
| disagreement_only | 0.200000 | 0.566667 |
| disagreement_only | 0.400000 | 0.616667 |
| disagreement_only | 0.600000 | 0.633333 |
| disagreement_only | 0.800000 | 0.641667 |
| disagreement_only | 1.000000 | 0.646667 |
| invariant_only | 0.050000 | 0.640000 |
| invariant_only | 0.100000 | 0.653333 |
| invariant_only | 0.200000 | 0.643333 |
| invariant_only | 0.400000 | 0.641667 |
| invariant_only | 0.600000 | 0.646667 |
| invariant_only | 0.800000 | 0.648333 |
| invariant_only | 1.000000 | 0.646667 |
| isotonic_calibrated_judge | 0.050000 | 0.653333 |
| isotonic_calibrated_judge | 0.100000 | 0.646667 |
| isotonic_calibrated_judge | 0.200000 | 0.646667 |
| isotonic_calibrated_judge | 0.400000 | 0.645000 |
| isotonic_calibrated_judge | 0.600000 | 0.651111 |
| isotonic_calibrated_judge | 0.800000 | 0.649167 |
| isotonic_calibrated_judge | 1.000000 | 0.646667 |
| logistic_calibrated_judge | 0.050000 | 0.506667 |
| logistic_calibrated_judge | 0.100000 | 0.526667 |
| logistic_calibrated_judge | 0.200000 | 0.580000 |
| logistic_calibrated_judge | 0.400000 | 0.616667 |
| logistic_calibrated_judge | 0.600000 | 0.633333 |
| logistic_calibrated_judge | 0.800000 | 0.641667 |
| logistic_calibrated_judge | 1.000000 | 0.646667 |
| oracle_error_rank | 0.050000 | 0.466667 |
| oracle_error_rank | 0.100000 | 0.480000 |
| oracle_error_rank | 0.200000 | 0.566667 |
| oracle_error_rank | 0.400000 | 0.616667 |
| oracle_error_rank | 0.600000 | 0.633333 |
| oracle_error_rank | 0.800000 | 0.641667 |
| oracle_error_rank | 1.000000 | 0.646667 |
| quantile_rule_judge | 0.050000 | 0.640000 |
| quantile_rule_judge | 0.100000 | 0.653333 |
| quantile_rule_judge | 0.200000 | 0.643333 |
| quantile_rule_judge | 0.400000 | 0.641667 |
| quantile_rule_judge | 0.600000 | 0.646667 |
| quantile_rule_judge | 0.800000 | 0.648333 |
| quantile_rule_judge | 1.000000 | 0.646667 |
| random_baseline | 0.050000 | 0.653333 |
| random_baseline | 0.100000 | 0.653333 |
| random_baseline | 0.200000 | 0.656667 |
| random_baseline | 0.400000 | 0.646667 |
| random_baseline | 0.600000 | 0.650000 |
| random_baseline | 0.800000 | 0.645833 |
| random_baseline | 1.000000 | 0.646667 |
| rank_normalized_linear | 0.050000 | 0.533333 |
| rank_normalized_linear | 0.100000 | 0.573333 |
| rank_normalized_linear | 0.200000 | 0.600000 |
| rank_normalized_linear | 0.400000 | 0.621667 |
| rank_normalized_linear | 0.600000 | 0.633333 |
| rank_normalized_linear | 0.800000 | 0.641667 |
| rank_normalized_linear | 1.000000 | 0.646667 |
| repair_only | 0.050000 | 0.666667 |
| repair_only | 0.100000 | 0.653333 |
| repair_only | 0.200000 | 0.653333 |

## Comparison against v0 combined_linear

| model_id | scenario_type | coverage | best_calibrated_judge | calibrated_far | combined_linear_far | beats_combined_linear |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| hold_last | normal_policy | 0.050000 | logistic_calibrated_judge | 0.200000 | 0.000000 | False |
| hold_last | normal_policy | 0.100000 | logistic_calibrated_judge | 0.300000 | 0.100000 | False |
| hold_last | normal_policy | 0.200000 | logistic_calibrated_judge | 0.550000 | 0.550000 | False |
| hold_last | normal_policy | 0.400000 | rank_normalized_linear | 0.775000 | 0.775000 | False |
| hold_last | normal_policy | 0.600000 | rank_normalized_linear | 0.850000 | 0.850000 | False |
| hold_last | normal_policy | 0.800000 | rank_normalized_linear | 0.887500 | 0.887500 | False |
| hold_last | normal_policy | 1.000000 | rank_normalized_linear | 0.910000 | 0.910000 | False |
| linear_narx | normal_policy | 0.050000 | rank_normalized_linear | 0.000000 | 0.000000 | False |
| linear_narx | normal_policy | 0.100000 | rank_normalized_linear | 0.000000 | 0.000000 | False |
| linear_narx | normal_policy | 0.200000 | rank_normalized_linear | 0.000000 | 0.000000 | False |
| linear_narx | normal_policy | 0.400000 | rank_normalized_linear | 0.000000 | 0.000000 | False |
| linear_narx | normal_policy | 0.600000 | rank_normalized_linear | 0.000000 | 0.000000 | False |
| linear_narx | normal_policy | 0.800000 | rank_normalized_linear | 0.000000 | 0.000000 | False |
| linear_narx | normal_policy | 1.000000 | rank_normalized_linear | 0.000000 | 0.000000 | False |
| mlp_state_space | normal_policy | 0.050000 | rank_normalized_linear | 0.000000 | 0.000000 | False |
| mlp_state_space | normal_policy | 0.100000 | rank_normalized_linear | 0.000000 | 0.000000 | False |
| mlp_state_space | normal_policy | 0.200000 | rank_normalized_linear | 0.000000 | 0.000000 | False |
| mlp_state_space | normal_policy | 0.400000 | rank_normalized_linear | 0.000000 | 0.000000 | False |
| mlp_state_space | normal_policy | 0.600000 | rank_normalized_linear | 0.000000 | 0.000000 | False |
| mlp_state_space | normal_policy | 0.800000 | rank_normalized_linear | 0.000000 | 0.000000 | False |
| mlp_state_space | normal_policy | 1.000000 | rank_normalized_linear | 0.000000 | 0.000000 | False |
| hold_last | held_out_action_magnitude | 0.050000 | conservative_low_coverage_judge | 0.000000 | 0.000000 | False |
| hold_last | held_out_action_magnitude | 0.100000 | conservative_low_coverage_judge | 0.200000 | 0.200000 | False |
| hold_last | held_out_action_magnitude | 0.200000 | logistic_calibrated_judge | 0.550000 | 0.550000 | False |
| hold_last | held_out_action_magnitude | 0.400000 | rank_normalized_linear | 0.775000 | 0.775000 | False |
| hold_last | held_out_action_magnitude | 0.600000 | rank_normalized_linear | 0.850000 | 0.850000 | False |
| hold_last | held_out_action_magnitude | 0.800000 | rank_normalized_linear | 0.887500 | 0.887500 | False |
| hold_last | held_out_action_magnitude | 1.000000 | rank_normalized_linear | 0.910000 | 0.910000 | False |
| linear_narx | held_out_action_magnitude | 0.050000 | rank_normalized_linear | 1.000000 | 1.000000 | False |
| linear_narx | held_out_action_magnitude | 0.100000 | rank_normalized_linear | 1.000000 | 1.000000 | False |

## Comparison against best single signal selected on calibration

| model_id | scenario_type | coverage | best_deployable_baseline | baseline_far | best_calibrated_judge | calibrated_far | margin |
| --- | --- | ---: | --- | ---: | --- | ---: | ---: |
| hold_last | normal_policy | 0.050000 | best_single_signal_selected_on_calibration | 1.000000 | logistic_calibrated_judge | 0.200000 | 0.800000 |
| hold_last | normal_policy | 0.100000 | best_single_signal_selected_on_calibration | 1.000000 | logistic_calibrated_judge | 0.300000 | 0.700000 |
| hold_last | normal_policy | 0.200000 | best_single_signal_selected_on_calibration | 0.950000 | logistic_calibrated_judge | 0.550000 | 0.400000 |
| hold_last | normal_policy | 0.400000 | best_single_signal_selected_on_calibration | 0.900000 | rank_normalized_linear | 0.775000 | 0.125000 |
| hold_last | normal_policy | 0.600000 | best_single_signal_selected_on_calibration | 0.916667 | rank_normalized_linear | 0.850000 | 0.066667 |
| hold_last | normal_policy | 0.800000 | best_single_signal_selected_on_calibration | 0.925000 | rank_normalized_linear | 0.887500 | 0.037500 |
| hold_last | normal_policy | 1.000000 | best_single_signal_selected_on_calibration | 0.910000 | rank_normalized_linear | 0.910000 | 0.000000 |
| linear_narx | normal_policy | 0.050000 | best_single_signal_selected_on_calibration | 0.000000 | rank_normalized_linear | 0.000000 | 0.000000 |
| linear_narx | normal_policy | 0.100000 | best_single_signal_selected_on_calibration | 0.000000 | rank_normalized_linear | 0.000000 | 0.000000 |
| linear_narx | normal_policy | 0.200000 | best_single_signal_selected_on_calibration | 0.000000 | rank_normalized_linear | 0.000000 | 0.000000 |
| linear_narx | normal_policy | 0.400000 | best_single_signal_selected_on_calibration | 0.000000 | rank_normalized_linear | 0.000000 | 0.000000 |
| linear_narx | normal_policy | 0.600000 | best_single_signal_selected_on_calibration | 0.000000 | rank_normalized_linear | 0.000000 | 0.000000 |
| linear_narx | normal_policy | 0.800000 | best_single_signal_selected_on_calibration | 0.000000 | rank_normalized_linear | 0.000000 | 0.000000 |
| linear_narx | normal_policy | 1.000000 | best_single_signal_selected_on_calibration | 0.000000 | rank_normalized_linear | 0.000000 | 0.000000 |
| mlp_state_space | normal_policy | 0.050000 | best_single_signal_selected_on_calibration | 0.000000 | rank_normalized_linear | 0.000000 | 0.000000 |
| mlp_state_space | normal_policy | 0.100000 | best_single_signal_selected_on_calibration | 0.000000 | rank_normalized_linear | 0.000000 | 0.000000 |
| mlp_state_space | normal_policy | 0.200000 | best_single_signal_selected_on_calibration | 0.000000 | rank_normalized_linear | 0.000000 | 0.000000 |
| mlp_state_space | normal_policy | 0.400000 | best_single_signal_selected_on_calibration | 0.000000 | rank_normalized_linear | 0.000000 | 0.000000 |
| mlp_state_space | normal_policy | 0.600000 | best_single_signal_selected_on_calibration | 0.000000 | rank_normalized_linear | 0.000000 | 0.000000 |
| mlp_state_space | normal_policy | 0.800000 | best_single_signal_selected_on_calibration | 0.000000 | rank_normalized_linear | 0.000000 | 0.000000 |
| mlp_state_space | normal_policy | 1.000000 | best_single_signal_selected_on_calibration | 0.000000 | rank_normalized_linear | 0.000000 | 0.000000 |
| hold_last | held_out_action_magnitude | 0.050000 | best_single_signal_selected_on_calibration | 0.600000 | conservative_low_coverage_judge | 0.000000 | 0.600000 |
| hold_last | held_out_action_magnitude | 0.100000 | best_single_signal_selected_on_calibration | 0.800000 | conservative_low_coverage_judge | 0.200000 | 0.600000 |
| hold_last | held_out_action_magnitude | 0.200000 | best_single_signal_selected_on_calibration | 0.850000 | logistic_calibrated_judge | 0.550000 | 0.300000 |
| hold_last | held_out_action_magnitude | 0.400000 | best_single_signal_selected_on_calibration | 0.900000 | rank_normalized_linear | 0.775000 | 0.125000 |
| hold_last | held_out_action_magnitude | 0.600000 | best_single_signal_selected_on_calibration | 0.916667 | rank_normalized_linear | 0.850000 | 0.066667 |
| hold_last | held_out_action_magnitude | 0.800000 | best_single_signal_selected_on_calibration | 0.912500 | rank_normalized_linear | 0.887500 | 0.025000 |
| hold_last | held_out_action_magnitude | 1.000000 | best_single_signal_selected_on_calibration | 0.910000 | rank_normalized_linear | 0.910000 | 0.000000 |
| linear_narx | held_out_action_magnitude | 0.050000 | best_single_signal_selected_on_calibration | 1.000000 | rank_normalized_linear | 1.000000 | 0.000000 |
| linear_narx | held_out_action_magnitude | 0.100000 | best_single_signal_selected_on_calibration | 1.000000 | rank_normalized_linear | 1.000000 | 0.000000 |

## Oracle diagnostic gap

| coverage_requested | oracle_far | best_real_far | oracle_gap |
| ---: | ---: | ---: | ---: |
| 0.050000 | 0.466667 | 0.466667 | 0.000000 |
| 0.100000 | 0.480000 | 0.486667 | 0.006667 |
| 0.200000 | 0.566667 | 0.566667 | 0.000000 |
| 0.400000 | 0.616667 | 0.616667 | 0.000000 |
| 0.600000 | 0.633333 | 0.633333 | 0.000000 |
| 0.800000 | 0.641667 | 0.641667 | 0.000000 |
| 1.000000 | 0.646667 | 0.646667 | 0.000000 |

## Unavailable judges

none

## Verdict

MIXED

## Explanation

Best calibrated judge: rank_normalized_linear. Low-coverage win rate versus the calibration-selected single-signal baseline: 0.200000. Expansion remains forbidden.

## Known failures

- calibrated judges did not establish a broad selective-simulation claim

## Reproduction command

```bash
python scripts/run_calibrated_judge.py --config configs/experiments/calibrated_two_tank.yaml --output results/calibrated_two_tank
```
