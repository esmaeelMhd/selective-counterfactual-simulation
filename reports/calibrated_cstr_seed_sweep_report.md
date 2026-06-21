# Calibrated CSTR Seed Sweep Report

## Command

```bash
python scripts/run_calibrated_seed_sweep.py --config /tmp/pytest-of-ismayil/pytest-147/test_tiny_cstr_seed_sweep_keep0/tiny_cstr.yaml --seeds 0 1 --output /tmp/pytest-of-ismayil/pytest-147/test_tiny_cstr_seed_sweep_keep0/seed_sweep
```

## Seeds

0, 1

## Per-seed verdict

| seed | verdict | best_calibrated_judge | low_coverage_win | leakage_detected |
| ---: | --- | --- | ---: | ---: |
| 0 | NO_IMPROVEMENT_OVER_SINGLE_SIGNAL | rank_normalized_linear | False | False |
| 1 | MIXED | rank_normalized_linear | False | False |

## Low-coverage aggregate

| coverage | win_rate_vs_calibration_selected_single_signal | mean_margin | std_margin |
| ---: | ---: | ---: | ---: |
| 0.050000 | 0.000000 | 0.000000 | 0.000000 |
| 0.100000 | 0.000000 | 0.000000 | 0.000000 |

## Judge robustness

| judge | win_count | mean_far | std_far |
| --- | ---: | ---: | ---: |
| best_single_signal_selected_on_calibration | 56 | 0.571429 | 0.499350 |
| calibration_selected_candidate_ranker | 56 | 0.571429 | 0.499350 |
| combined_linear | 56 | 0.571429 | 0.499350 |
| conservative_low_coverage_judge | 56 | 0.571429 | 0.499350 |
| disagreement_only | 56 | 0.571429 | 0.499350 |
| invariant_only | 56 | 0.571429 | 0.499350 |
| isotonic_calibrated_judge | 56 | 0.571429 | 0.499350 |
| logistic_calibrated_judge | 56 | 0.571429 | 0.499350 |
| oracle_error_rank | 56 | 0.535714 | 0.503236 |
| quantile_rule_judge | 56 | 0.571429 | 0.499350 |
| random_baseline | 56 | 0.535714 | 0.503236 |
| rank_normalized_linear | 56 | 0.571429 | 0.499350 |
| repair_only | 56 | 0.571429 | 0.499350 |
| support_only | 56 | 0.571429 | 0.499350 |
| uncertainty_only | 56 | 0.571429 | 0.499350 |

## Failure cases

none

## Verdict

NO_ROBUST_IMPROVEMENT

## Explanation

Calibrated low-coverage wins appeared in 0 of 2 seeds.
