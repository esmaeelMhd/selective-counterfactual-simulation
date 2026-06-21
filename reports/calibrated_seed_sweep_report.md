# Calibrated Judge Seed Sweep Report

## Command

```bash
python scripts/run_calibrated_seed_sweep.py --config /tmp/pytest-of-ismayil/pytest-147/test_tiny_calibrated_seed_swee0/tiny.yaml --seeds 0 1 --output /tmp/pytest-of-ismayil/pytest-147/test_tiny_calibrated_seed_swee0/seed_sweep
```

## Seeds

0, 1

## Per-seed verdict

| seed | verdict | best_calibrated_judge | low_coverage_win | leakage_detected |
| ---: | --- | --- | ---: | ---: |
| 0 | SUPPORTED_LOW_COVERAGE | rank_normalized_linear | True | False |
| 1 | SUPPORTED_LOW_COVERAGE | rank_normalized_linear | True | False |

## Low-coverage aggregate

| coverage | win_rate_vs_calibration_selected_single_signal | mean_margin | std_margin |
| ---: | ---: | ---: | ---: |
| 0.050000 | 1.000000 | 0.200000 | 0.000000 |
| 0.100000 | 1.000000 | 0.200000 | 0.000000 |

## Judge robustness

| judge | win_count | mean_far | std_far |
| --- | ---: | ---: | ---: |
| best_single_signal_selected_on_calibration | 40 | 0.500000 | 0.506370 |
| calibration_selected_candidate_ranker | 40 | 0.350000 | 0.483046 |
| combined_linear | 40 | 0.400000 | 0.496139 |
| conservative_low_coverage_judge | 40 | 0.450000 | 0.503831 |
| disagreement_only | 40 | 0.350000 | 0.483046 |
| invariant_only | 40 | 0.500000 | 0.506370 |
| isotonic_calibrated_judge | 40 | 0.550000 | 0.503831 |
| logistic_calibrated_judge | 40 | 0.350000 | 0.483046 |
| oracle_error_rank | 40 | 0.200000 | 0.405096 |
| quantile_rule_judge | 40 | 0.500000 | 0.506370 |
| random_baseline | 40 | 0.550000 | 0.503831 |
| rank_normalized_linear | 40 | 0.450000 | 0.503831 |
| repair_only | 40 | 0.550000 | 0.503831 |
| support_only | 40 | 0.600000 | 0.496139 |
| uncertainty_only | 40 | 0.550000 | 0.503831 |

## Failure cases

none

## Verdict

ROBUST_LOW_COVERAGE

## Explanation

Calibrated low-coverage wins appeared in 2 of 2 seeds.
