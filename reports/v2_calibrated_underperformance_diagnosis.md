# v2 Calibrated Underperformance Diagnosis

## Verdict

UNDERPERFORMANCE_DIAGNOSED

## Scope

This diagnosis reads existing frozen v2 artifacts only. It does not change the v2 protocol, does not rerun model training, and does not upgrade any claim.

## Primary Finding

The primary calibrated candidate underperforms because it is compared against a moving strongest-baseline envelope, and because event-risk targets expose failures that the calibrated ranker does not handle robustly.

## Primary vs Baseline

| system_id | badness_target | mean_primary_far | mean_baseline_far | mean_absolute_margin | losing_row_rate |
| --- | --- | ---: | ---: | ---: | ---: |
| cstr | bad_event | 0.045357 | 0.000000 | -0.045357 | 0.240000 |
| cstr | bad_rmse | 0.178214 | 0.176905 | -0.001310 | 0.006667 |
| cstr | bad_rmse_or_event | 0.178214 | 0.176488 | -0.001726 | 0.008333 |
| heat_exchanger | bad_event | 0.220000 | 0.156786 | -0.063214 | 0.220000 |
| heat_exchanger | bad_rmse | 0.228750 | 0.226786 | -0.001964 | 0.008333 |
| heat_exchanger | bad_rmse_or_event | 0.236726 | 0.232083 | -0.004643 | 0.023333 |
| two_tank | bad_event | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| two_tank | bad_rmse | 0.105556 | 0.061778 | -0.043778 | 0.118333 |
| two_tank | bad_rmse_or_event | 0.105556 | 0.061778 | -0.043778 | 0.118333 |

## Baseline Winner Counts

| system_id | badness_target | baseline_judge | winner_count |
| --- | --- | --- | ---: |
| cstr | bad_event | conformal_risk_threshold | 77 |
| cstr | bad_event | disagreement_only | 11 |
| cstr | bad_event | invariant_only | 11 |
| cstr | bad_event | learned_error_classifier | 1 |
| cstr | bad_rmse | conformal_risk_threshold | 596 |
| cstr | bad_rmse | learned_error_classifier | 2 |
| cstr | bad_rmse | random_baseline | 2 |
| cstr | bad_rmse_or_event | conformal_risk_threshold | 595 |
| cstr | bad_rmse_or_event | learned_error_classifier | 3 |
| cstr | bad_rmse_or_event | random_baseline | 2 |
| heat_exchanger | bad_event | conformal_risk_threshold | 82 |
| heat_exchanger | bad_event | learned_error_classifier | 8 |
| heat_exchanger | bad_event | disagreement_only | 5 |
| heat_exchanger | bad_event | random_baseline | 5 |
| heat_exchanger | bad_rmse | conformal_risk_threshold | 595 |
| heat_exchanger | bad_rmse | repair_only | 4 |
| heat_exchanger | bad_rmse | learned_error_classifier | 1 |
| heat_exchanger | bad_rmse_or_event | conformal_risk_threshold | 588 |

## Best Judges by FAR

| system_id | badness_target | judge_id | mean_far | rank |
| --- | --- | --- | ---: | ---: |
| cstr | bad_event | oracle_error_rank | 0.000000 | 1.000000 |
| cstr | bad_event | invariant_only | 0.001429 | 2.000000 |
| cstr | bad_event | support_only | 0.009286 | 3.000000 |
| cstr | bad_event | rank_normalized_linear | 0.010714 | 4.000000 |
| cstr | bad_event | repair_only | 0.017500 | 5.000000 |
| cstr | bad_rmse | oracle_error_rank | 0.136964 | 1.000000 |
| cstr | bad_rmse | isotonic_calibrated_judge | 0.178095 | 2.000000 |
| cstr | bad_rmse | best_single_signal_selected_on_calibration | 0.178214 | 3.000000 |
| cstr | bad_rmse | calibration_selected_candidate_ranker | 0.178214 | 4.000000 |
| cstr | bad_rmse | conformal_risk_threshold | 0.178214 | 5.000000 |
| cstr | bad_rmse_or_event | oracle_error_rank | 0.136964 | 1.000000 |
| cstr | bad_rmse_or_event | isotonic_calibrated_judge | 0.178095 | 2.000000 |
| cstr | bad_rmse_or_event | best_single_signal_selected_on_calibration | 0.178214 | 3.000000 |
| cstr | bad_rmse_or_event | calibration_selected_candidate_ranker | 0.178214 | 4.000000 |
| cstr | bad_rmse_or_event | conformal_risk_threshold | 0.178214 | 5.000000 |
| heat_exchanger | bad_event | oracle_error_rank | 0.005714 | 1.000000 |
| heat_exchanger | bad_event | disagreement_only | 0.200000 | 2.000000 |
| heat_exchanger | bad_event | ensemble_disagreement_threshold | 0.200000 | 3.000000 |
| heat_exchanger | bad_event | conservative_low_coverage_judge | 0.201429 | 4.000000 |
| heat_exchanger | bad_event | rank_normalized_linear | 0.206786 | 5.000000 |
| heat_exchanger | bad_rmse | oracle_error_rank | 0.221845 | 1.000000 |
| heat_exchanger | bad_rmse | best_single_signal_selected_on_calibration | 0.228750 | 2.000000 |
| heat_exchanger | bad_rmse | calibration_selected_candidate_ranker | 0.228750 | 3.000000 |
| heat_exchanger | bad_rmse | conformal_risk_threshold | 0.228750 | 4.000000 |
| heat_exchanger | bad_rmse | invariant_only | 0.228750 | 5.000000 |
| heat_exchanger | bad_rmse_or_event | oracle_error_rank | 0.223214 | 1.000000 |
| heat_exchanger | bad_rmse_or_event | best_single_signal_selected_on_calibration | 0.236071 | 2.000000 |
| heat_exchanger | bad_rmse_or_event | conformal_risk_threshold | 0.236071 | 3.000000 |
| heat_exchanger | bad_rmse_or_event | invariant_only | 0.236071 | 4.000000 |
| heat_exchanger | bad_rmse_or_event | quantile_rule_judge | 0.236071 | 5.000000 |

## Label Balance

| system_id | badness_target | row_count | bad_rate | event_mismatch_rate | primary_unique_scores |
| --- | --- | ---: | ---: | ---: | ---: |
| cstr | bad_event | 3500 | 0.082286 | 0.082286 | 869 |
| cstr | bad_rmse | 21000 | 0.635762 | 0.082286 | 3500 |
| cstr | bad_rmse_or_event | 21000 | 0.648857 | 0.082286 | 3500 |
| heat_exchanger | bad_event | 3500 | 0.220000 | 0.220000 | 3142 |
| heat_exchanger | bad_rmse | 21000 | 0.707857 | 0.220000 | 3500 |
| heat_exchanger | bad_rmse_or_event | 21000 | 0.715619 | 0.220000 | 3850 |
| two_tank | bad_event | 2500 | 0.000000 | 0.000000 | 1 |
| two_tank | bad_rmse | 15000 | 0.426667 | 0.000000 | 8383 |
| two_tank | bad_rmse_or_event | 15000 | 0.426667 | 0.000000 | 8383 |

## Root Causes

1. The current `baseline_far` is a row-wise envelope over baseline judges. That is stricter than comparing against one fixed deployable baseline and makes small negative margins likely even when the primary candidate ties a baseline on average.
2. `conformal_risk_threshold` is the dominant baseline winner, showing that simple oriented threshold rules often beat the calibrated candidate at low coverage.
3. Event-risk is the clearest failure mode: CSTR and heat_exchanger event targets have negative margins, while TwoTank event labels are degenerate.
4. The primary candidate is not the best calibrated-family member on TwoTank RMSE; learned/logistic scores rank that target better in the current artifacts.

## Claim Impact

The v2 decision remains `NO_METHOD_CLAIM_BENCHMARK_ONLY`. This diagnosis does not support a claim upgrade.

## Recommended Next Action

First produce a calibration-selected fixed-baseline comparison beside the current row-wise strongest-baseline envelope. Then run an event-risk-specific failure analysis before changing any judge.
