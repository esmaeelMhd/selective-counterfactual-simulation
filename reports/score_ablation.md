# Score Ablation and Calibration Report

## Question

Did combined_linear fail because of bad combination logic?

## Ablation results

| score | false_accept_rate | win_rate_vs_best_simple | verdict |
| --- | ---: | ---: | --- |
| combined_linear | 0.437500 | 0.000000 | NOT_SUPPORTED |
| combined_without_support | 0.437500 | 0.000000 | NOT_SUPPORTED |
| combined_without_uncertainty | 0.437500 | 0.000000 | NOT_SUPPORTED |
| combined_without_disagreement | 0.437500 | 0.000000 | NOT_SUPPORTED |
| combined_without_invariant | 0.437500 | 0.000000 | NOT_SUPPORTED |
| combined_without_repair | 0.437500 | 0.000000 | NOT_SUPPORTED |
| rank_normalized_combined | 0.437500 | 0.000000 | NOT_SUPPORTED |
| logistic_error_classifier | 0.437500 | 0.000000 | NOT_SUPPORTED |
| isotonic_calibrated_score | 0.437500 | 0.000000 | NOT_SUPPORTED |

## Signal removal effect

| removed_signal | delta_vs_combined_linear |
| --- | ---: |
| support | 0.000000 |
| uncertainty | 0.000000 |
| disagreement | 0.000000 |
| invariant | 0.000000 |
| repair | 0.000000 |

## Learned calibration result

| method | validation_scheme | win_rate | caveat |
| --- | --- | ---: | --- |
| logistic_error_classifier | grouped cross-validation by scenario_id | 0.000000 | grouped cross-validation by scenario_id |
| isotonic_calibrated_score | grouped cross-validation by scenario_id | 0.000000 | grouped cross-validation by scenario_id |

## Signals that hurt the combined score

none

## Signals that help the combined score

none

## Interpretation

Best ablation/calibration win rate was 0.000000; learned scores used grouped validation, not same-row train/test evaluation.

## Verdict

SIGNAL_PROBLEM
