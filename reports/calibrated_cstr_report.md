# Calibrated CSTR Refusal Judge Report

## Protocol lock

docs/calibrated_protocol_lock_v1.md freezes candidate judges, baseline judges, signal columns, models, coverage grid, threshold grid, seed rules, stress rules, and decision rules before CSTR evidence.



## Research question

Can a calibrated refusal judge identify a low-coverage subset of counterfactual rollouts with lower false-accept risk than the strongest calibration-selected single signal?

## Strict leakage statement

Simulator models are fit on `model_train`; calibrated judges are selected or fit on `judge_calibration_*`; final risk-coverage is computed on `judge_test_*`. Test labels are not used during judge fitting. Oracle is diagnostic only.

CSTR sanity must be `VALID_CSTR_BENCHMARK` before the full `calibrated_cstr` evidence run reports claim evidence.


## Data splits

combined_feed_and_cooling_shift, cooling_step_change, feed_concentration_spike, feed_temperature_spike, id, reaction_rate_shift, unsafe_temperature_event

## Models

hold_last, linear_narx, mlp_state_space

## Signals

support_distance, uncertainty_score, disagreement_score, invariant_residual, repair_amount

## Judges

support_only, uncertainty_only, disagreement_only, invariant_only, repair_only, combined_linear, best_single_signal_selected_on_calibration, rank_normalized_linear, logistic_calibrated_judge, isotonic_calibrated_judge, quantile_rule_judge, conservative_low_coverage_judge, calibration_selected_candidate_ranker, random_baseline, oracle_error_rank

## Calibration provenance

| judge_id | available | selected_signal_if_any | used_test_labels_during_fit |
| --- | ---: | --- | ---: |
| best_single_signal_selected_on_calibration | True | invariant_residual | False |
| rank_normalized_linear | True | none | False |
| calibration_selected_candidate_ranker | True | none | False |
| logistic_calibrated_judge | True | none | False |
| isotonic_calibrated_judge | True | invariant_residual | False |
| quantile_rule_judge | True | invariant_residual | False |
| conservative_low_coverage_judge | True | none | False |

## Low-coverage result

| coverage | best_deployable_baseline | baseline_far | best_calibrated_judge | calibrated_far | margin |
| ---: | --- | ---: | --- | ---: | ---: |
| 0.050000 | best_single_signal_selected_on_calibration | 0.666667 | rank_normalized_linear | 0.628571 | 0.038095 |
| 0.100000 | best_single_signal_selected_on_calibration | 0.666667 | rank_normalized_linear | 0.628571 | 0.038095 |

## Full risk-coverage result

| judge_id | coverage_requested | false_accept_rate |
| --- | ---: | ---: |
| best_single_signal_selected_on_calibration | 0.050000 | 0.666667 |
| best_single_signal_selected_on_calibration | 0.100000 | 0.666667 |
| best_single_signal_selected_on_calibration | 0.200000 | 0.645238 |
| best_single_signal_selected_on_calibration | 0.400000 | 0.647619 |
| best_single_signal_selected_on_calibration | 0.600000 | 0.668254 |
| best_single_signal_selected_on_calibration | 0.800000 | 0.691071 |
| best_single_signal_selected_on_calibration | 1.000000 | 0.720476 |
| calibration_selected_candidate_ranker | 0.050000 | 0.638095 |
| calibration_selected_candidate_ranker | 0.100000 | 0.657143 |
| calibration_selected_candidate_ranker | 0.200000 | 0.652381 |
| calibration_selected_candidate_ranker | 0.400000 | 0.655952 |
| calibration_selected_candidate_ranker | 0.600000 | 0.678571 |
| calibration_selected_candidate_ranker | 0.800000 | 0.700000 |
| calibration_selected_candidate_ranker | 1.000000 | 0.720476 |
| combined_linear | 0.050000 | 0.685714 |
| combined_linear | 0.100000 | 0.676190 |
| combined_linear | 0.200000 | 0.690476 |
| combined_linear | 0.400000 | 0.698810 |
| combined_linear | 0.600000 | 0.703175 |
| combined_linear | 0.800000 | 0.707738 |
| combined_linear | 1.000000 | 0.720476 |
| conservative_low_coverage_judge | 0.050000 | 0.657143 |
| conservative_low_coverage_judge | 0.100000 | 0.652381 |
| conservative_low_coverage_judge | 0.200000 | 0.669048 |
| conservative_low_coverage_judge | 0.400000 | 0.697619 |
| conservative_low_coverage_judge | 0.600000 | 0.703175 |
| conservative_low_coverage_judge | 0.800000 | 0.707143 |
| conservative_low_coverage_judge | 1.000000 | 0.720476 |
| disagreement_only | 0.050000 | 0.723810 |
| disagreement_only | 0.100000 | 0.719048 |
| disagreement_only | 0.200000 | 0.709524 |
| disagreement_only | 0.400000 | 0.701190 |
| disagreement_only | 0.600000 | 0.708730 |
| disagreement_only | 0.800000 | 0.716071 |
| disagreement_only | 1.000000 | 0.720476 |
| invariant_only | 0.050000 | 0.666667 |
| invariant_only | 0.100000 | 0.666667 |
| invariant_only | 0.200000 | 0.645238 |
| invariant_only | 0.400000 | 0.647619 |
| invariant_only | 0.600000 | 0.668254 |
| invariant_only | 0.800000 | 0.691071 |
| invariant_only | 1.000000 | 0.720476 |
| isotonic_calibrated_judge | 0.050000 | 0.628571 |
| isotonic_calibrated_judge | 0.100000 | 0.628571 |
| isotonic_calibrated_judge | 0.200000 | 0.640476 |
| isotonic_calibrated_judge | 0.400000 | 0.646429 |
| isotonic_calibrated_judge | 0.600000 | 0.670635 |
| isotonic_calibrated_judge | 0.800000 | 0.694643 |
| isotonic_calibrated_judge | 1.000000 | 0.720476 |
| logistic_calibrated_judge | 0.050000 | 0.647619 |
| logistic_calibrated_judge | 0.100000 | 0.642857 |
| logistic_calibrated_judge | 0.200000 | 0.664286 |
| logistic_calibrated_judge | 0.400000 | 0.686905 |
| logistic_calibrated_judge | 0.600000 | 0.697619 |
| logistic_calibrated_judge | 0.800000 | 0.701786 |
| logistic_calibrated_judge | 1.000000 | 0.720476 |
| oracle_error_rank | 0.050000 | 0.619048 |
| oracle_error_rank | 0.100000 | 0.619048 |
| oracle_error_rank | 0.200000 | 0.619048 |
| oracle_error_rank | 0.400000 | 0.626190 |
| oracle_error_rank | 0.600000 | 0.663492 |
| oracle_error_rank | 0.800000 | 0.688095 |
| oracle_error_rank | 1.000000 | 0.720476 |
| quantile_rule_judge | 0.050000 | 0.666667 |
| quantile_rule_judge | 0.100000 | 0.666667 |
| quantile_rule_judge | 0.200000 | 0.645238 |
| quantile_rule_judge | 0.400000 | 0.647619 |
| quantile_rule_judge | 0.600000 | 0.668254 |
| quantile_rule_judge | 0.800000 | 0.691071 |
| quantile_rule_judge | 1.000000 | 0.720476 |
| random_baseline | 0.050000 | 0.695238 |
| random_baseline | 0.100000 | 0.704762 |
| random_baseline | 0.200000 | 0.723810 |
| random_baseline | 0.400000 | 0.726190 |
| random_baseline | 0.600000 | 0.719048 |
| random_baseline | 0.800000 | 0.720833 |
| random_baseline | 1.000000 | 0.720476 |
| rank_normalized_linear | 0.050000 | 0.666667 |
| rank_normalized_linear | 0.100000 | 0.666667 |
| rank_normalized_linear | 0.200000 | 0.645238 |

## Comparison against v0 combined_linear

| model_id | scenario_type | coverage | best_calibrated_judge | calibrated_far | combined_linear_far | beats_combined_linear |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| hold_last | combined_feed_and_cooling_shift | 0.050000 | rank_normalized_linear | 1.000000 | 1.000000 | False |
| hold_last | combined_feed_and_cooling_shift | 0.100000 | rank_normalized_linear | 1.000000 | 1.000000 | False |
| hold_last | combined_feed_and_cooling_shift | 0.200000 | rank_normalized_linear | 1.000000 | 1.000000 | False |
| hold_last | combined_feed_and_cooling_shift | 0.400000 | rank_normalized_linear | 1.000000 | 1.000000 | False |
| hold_last | combined_feed_and_cooling_shift | 0.600000 | rank_normalized_linear | 1.000000 | 1.000000 | False |
| hold_last | combined_feed_and_cooling_shift | 0.800000 | rank_normalized_linear | 1.000000 | 1.000000 | False |
| hold_last | combined_feed_and_cooling_shift | 1.000000 | rank_normalized_linear | 1.000000 | 1.000000 | False |
| linear_narx | combined_feed_and_cooling_shift | 0.050000 | rank_normalized_linear | 0.000000 | 0.200000 | True |
| linear_narx | combined_feed_and_cooling_shift | 0.100000 | rank_normalized_linear | 0.000000 | 0.200000 | True |
| linear_narx | combined_feed_and_cooling_shift | 0.200000 | rank_normalized_linear | 0.000000 | 0.350000 | True |
| linear_narx | combined_feed_and_cooling_shift | 0.400000 | calibration_selected_candidate_ranker | 0.200000 | 0.425000 | True |
| linear_narx | combined_feed_and_cooling_shift | 0.600000 | calibration_selected_candidate_ranker | 0.450000 | 0.516667 | True |
| linear_narx | combined_feed_and_cooling_shift | 0.800000 | rank_normalized_linear | 0.587500 | 0.612500 | True |
| linear_narx | combined_feed_and_cooling_shift | 1.000000 | rank_normalized_linear | 0.660000 | 0.660000 | False |
| mlp_state_space | combined_feed_and_cooling_shift | 0.050000 | rank_normalized_linear | 1.000000 | 1.000000 | False |
| mlp_state_space | combined_feed_and_cooling_shift | 0.100000 | rank_normalized_linear | 1.000000 | 1.000000 | False |
| mlp_state_space | combined_feed_and_cooling_shift | 0.200000 | rank_normalized_linear | 1.000000 | 1.000000 | False |
| mlp_state_space | combined_feed_and_cooling_shift | 0.400000 | rank_normalized_linear | 1.000000 | 1.000000 | False |
| mlp_state_space | combined_feed_and_cooling_shift | 0.600000 | rank_normalized_linear | 1.000000 | 1.000000 | False |
| mlp_state_space | combined_feed_and_cooling_shift | 0.800000 | rank_normalized_linear | 1.000000 | 1.000000 | False |
| mlp_state_space | combined_feed_and_cooling_shift | 1.000000 | rank_normalized_linear | 1.000000 | 1.000000 | False |
| hold_last | cooling_step_change | 0.050000 | rank_normalized_linear | 1.000000 | 1.000000 | False |
| hold_last | cooling_step_change | 0.100000 | rank_normalized_linear | 1.000000 | 1.000000 | False |
| hold_last | cooling_step_change | 0.200000 | rank_normalized_linear | 1.000000 | 1.000000 | False |
| hold_last | cooling_step_change | 0.400000 | rank_normalized_linear | 1.000000 | 1.000000 | False |
| hold_last | cooling_step_change | 0.600000 | rank_normalized_linear | 1.000000 | 1.000000 | False |
| hold_last | cooling_step_change | 0.800000 | rank_normalized_linear | 1.000000 | 1.000000 | False |
| hold_last | cooling_step_change | 1.000000 | rank_normalized_linear | 1.000000 | 1.000000 | False |
| linear_narx | cooling_step_change | 0.050000 | rank_normalized_linear | 0.000000 | 0.200000 | True |
| linear_narx | cooling_step_change | 0.100000 | rank_normalized_linear | 0.000000 | 0.200000 | True |

## Comparison against TwoTank result

TwoTank single-run verdict: SUPPORTED_LOW_COVERAGE; best calibrated judge: calibration_selected_candidate_ranker.


## Comparison against best single signal selected on calibration

| model_id | scenario_type | coverage | best_deployable_baseline | baseline_far | best_calibrated_judge | calibrated_far | margin |
| --- | --- | ---: | --- | ---: | --- | ---: | ---: |
| hold_last | combined_feed_and_cooling_shift | 0.050000 | best_single_signal_selected_on_calibration | 1.000000 | rank_normalized_linear | 1.000000 | 0.000000 |
| hold_last | combined_feed_and_cooling_shift | 0.100000 | best_single_signal_selected_on_calibration | 1.000000 | rank_normalized_linear | 1.000000 | 0.000000 |
| hold_last | combined_feed_and_cooling_shift | 0.200000 | best_single_signal_selected_on_calibration | 1.000000 | rank_normalized_linear | 1.000000 | 0.000000 |
| hold_last | combined_feed_and_cooling_shift | 0.400000 | best_single_signal_selected_on_calibration | 1.000000 | rank_normalized_linear | 1.000000 | 0.000000 |
| hold_last | combined_feed_and_cooling_shift | 0.600000 | best_single_signal_selected_on_calibration | 1.000000 | rank_normalized_linear | 1.000000 | 0.000000 |
| hold_last | combined_feed_and_cooling_shift | 0.800000 | best_single_signal_selected_on_calibration | 1.000000 | rank_normalized_linear | 1.000000 | 0.000000 |
| hold_last | combined_feed_and_cooling_shift | 1.000000 | best_single_signal_selected_on_calibration | 1.000000 | rank_normalized_linear | 1.000000 | 0.000000 |
| linear_narx | combined_feed_and_cooling_shift | 0.050000 | best_single_signal_selected_on_calibration | 0.000000 | rank_normalized_linear | 0.000000 | 0.000000 |
| linear_narx | combined_feed_and_cooling_shift | 0.100000 | best_single_signal_selected_on_calibration | 0.000000 | rank_normalized_linear | 0.000000 | 0.000000 |
| linear_narx | combined_feed_and_cooling_shift | 0.200000 | best_single_signal_selected_on_calibration | 0.000000 | rank_normalized_linear | 0.000000 | 0.000000 |
| linear_narx | combined_feed_and_cooling_shift | 0.400000 | best_single_signal_selected_on_calibration | 0.300000 | calibration_selected_candidate_ranker | 0.200000 | 0.100000 |
| linear_narx | combined_feed_and_cooling_shift | 0.600000 | best_single_signal_selected_on_calibration | 0.483333 | calibration_selected_candidate_ranker | 0.450000 | 0.033333 |
| linear_narx | combined_feed_and_cooling_shift | 0.800000 | best_single_signal_selected_on_calibration | 0.587500 | rank_normalized_linear | 0.587500 | 0.000000 |
| linear_narx | combined_feed_and_cooling_shift | 1.000000 | best_single_signal_selected_on_calibration | 0.660000 | rank_normalized_linear | 0.660000 | 0.000000 |
| mlp_state_space | combined_feed_and_cooling_shift | 0.050000 | best_single_signal_selected_on_calibration | 1.000000 | rank_normalized_linear | 1.000000 | 0.000000 |
| mlp_state_space | combined_feed_and_cooling_shift | 0.100000 | best_single_signal_selected_on_calibration | 1.000000 | rank_normalized_linear | 1.000000 | 0.000000 |
| mlp_state_space | combined_feed_and_cooling_shift | 0.200000 | best_single_signal_selected_on_calibration | 1.000000 | rank_normalized_linear | 1.000000 | 0.000000 |
| mlp_state_space | combined_feed_and_cooling_shift | 0.400000 | best_single_signal_selected_on_calibration | 1.000000 | rank_normalized_linear | 1.000000 | 0.000000 |
| mlp_state_space | combined_feed_and_cooling_shift | 0.600000 | best_single_signal_selected_on_calibration | 1.000000 | rank_normalized_linear | 1.000000 | 0.000000 |
| mlp_state_space | combined_feed_and_cooling_shift | 0.800000 | best_single_signal_selected_on_calibration | 1.000000 | rank_normalized_linear | 1.000000 | 0.000000 |
| mlp_state_space | combined_feed_and_cooling_shift | 1.000000 | best_single_signal_selected_on_calibration | 1.000000 | rank_normalized_linear | 1.000000 | 0.000000 |
| hold_last | cooling_step_change | 0.050000 | best_single_signal_selected_on_calibration | 1.000000 | rank_normalized_linear | 1.000000 | 0.000000 |
| hold_last | cooling_step_change | 0.100000 | best_single_signal_selected_on_calibration | 1.000000 | rank_normalized_linear | 1.000000 | 0.000000 |
| hold_last | cooling_step_change | 0.200000 | best_single_signal_selected_on_calibration | 1.000000 | rank_normalized_linear | 1.000000 | 0.000000 |
| hold_last | cooling_step_change | 0.400000 | best_single_signal_selected_on_calibration | 1.000000 | rank_normalized_linear | 1.000000 | 0.000000 |
| hold_last | cooling_step_change | 0.600000 | best_single_signal_selected_on_calibration | 1.000000 | rank_normalized_linear | 1.000000 | 0.000000 |
| hold_last | cooling_step_change | 0.800000 | best_single_signal_selected_on_calibration | 1.000000 | rank_normalized_linear | 1.000000 | 0.000000 |
| hold_last | cooling_step_change | 1.000000 | best_single_signal_selected_on_calibration | 1.000000 | rank_normalized_linear | 1.000000 | 0.000000 |
| linear_narx | cooling_step_change | 0.050000 | best_single_signal_selected_on_calibration | 0.000000 | rank_normalized_linear | 0.000000 | 0.000000 |
| linear_narx | cooling_step_change | 0.100000 | best_single_signal_selected_on_calibration | 0.000000 | rank_normalized_linear | 0.000000 | 0.000000 |

## Oracle diagnostic gap

| coverage_requested | oracle_far | best_real_far | oracle_gap |
| ---: | ---: | ---: | ---: |
| 0.050000 | 0.619048 | 0.628571 | 0.009524 |
| 0.100000 | 0.619048 | 0.628571 | 0.009524 |
| 0.200000 | 0.619048 | 0.640476 | 0.021429 |
| 0.400000 | 0.626190 | 0.646429 | 0.020238 |
| 0.600000 | 0.663492 | 0.668254 | 0.004762 |
| 0.800000 | 0.688095 | 0.691071 | 0.002976 |
| 1.000000 | 0.720476 | 0.720476 | 0.000000 |

## Unavailable judges

none

## Verdict

SUPPORTED_LOW_COVERAGE

## Explanation

Best calibrated judge: rank_normalized_linear. Low-coverage win rate versus the calibration-selected single-signal baseline: 0.047619. Expansion remains forbidden.

## Known failures

- none

## Reproduction command

```bash
python scripts/run_calibrated_judge.py --config configs/experiments/calibrated_cstr.yaml --output results/calibrated_cstr
```
