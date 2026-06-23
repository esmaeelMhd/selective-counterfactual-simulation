# v2 Event-Risk Failure Diagnosis and Fix

## Verdict

EVENT_GUARD_REDUCES_EVENT_FALSE_ACCEPTS_NO_CLAIM_UPGRADE

## Diagnosis

The event target failure is not uniform across systems. CSTR event false accepts are best ranked by invariant/support-like scores. Heat-exchanger event false accepts are best ranked by disagreement/conservative-like scores. The primary calibrated ranker averages across targets and does not reliably preserve those event-specific rankings.

## Implemented Fix

`event_guarded_invariant_disagreement_support` uses existing risk scores only:

- `risk_invariant_only`
- `risk_disagreement_only`
- `risk_support_only`

The score is the maximum calibration-percentile rank across those inputs. Lower score means a scenario is accepted only when all guarded event-relevant signals are low relative to calibration support.

## No-Leakage Controls

- calibration rows are used only for percentile normalization;
- test labels are used only for evaluation;
- no new systems, models, or raw signals are introduced;
- no scientific claim is upgraded.

## Event Guard Comparison

| system_id | event_guard_far | primary_far | fair_baseline_far | row_wise_envelope_far | improvement_vs_primary | margin_vs_fair_baseline | margin_vs_row_wise_envelope |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cstr | 0.010714 | 0.045357 | 0.046429 | 0.000000 | 0.034643 | 0.035714 | -0.010714 |
| heat_exchanger | 0.200000 | 0.220000 | 0.208214 | 0.156786 | 0.020000 | 0.008214 | -0.043214 |
| two_tank | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## Candidate Signal Diagnosis

| system_id | judge_id | test_event_far_mean |
| --- | --- | ---: |
| cstr | invariant_only | 0.001429 |
| cstr | support_only | 0.009286 |
| cstr | event_guarded_invariant_disagreement_support | 0.010714 |
| cstr | rank_normalized_linear | 0.010714 |
| cstr | repair_only | 0.017500 |
| cstr | learned_error_classifier | 0.020000 |
| cstr | logistic_calibrated_judge | 0.020000 |
| cstr | conservative_low_coverage_judge | 0.020000 |
| cstr | uncertainty_only | 0.042500 |
| cstr | best_single_signal_selected_on_calibration | 0.045357 |
| cstr | calibration_selected_candidate_ranker | 0.045357 |
| cstr | quantile_rule_judge | 0.045357 |
| cstr | conformal_risk_threshold | 0.046429 |
| cstr | isotonic_calibrated_judge | 0.057143 |
| cstr | random_baseline | 0.091786 |
| cstr | disagreement_only | 0.152143 |
| cstr | ensemble_disagreement_threshold | 0.152143 |
| heat_exchanger | disagreement_only | 0.200000 |
| heat_exchanger | ensemble_disagreement_threshold | 0.200000 |
| heat_exchanger | event_guarded_invariant_disagreement_support | 0.200000 |
| heat_exchanger | conservative_low_coverage_judge | 0.201429 |
| heat_exchanger | rank_normalized_linear | 0.206786 |
| heat_exchanger | support_only | 0.207143 |
| heat_exchanger | conformal_risk_threshold | 0.208214 |

## Claim Impact

This is a targeted event-risk repair candidate. It does not support a general calibrated-refusal claim. The diagnostic row-wise envelope remains stricter than the implemented guard.
