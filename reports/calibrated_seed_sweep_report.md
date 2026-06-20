# Calibrated Judge Seed Sweep Report

## Command

```bash
python scripts/run_calibrated_seed_sweep.py --config configs/experiments/calibrated_two_tank.yaml --seeds 0 1 2 3 4 5 6 7 8 9 --output results/calibrated_seed_sweep_two_tank
```

## Seeds

0, 1, 2, 3, 4, 5, 6, 7, 8, 9

## Per-seed verdict

| seed | verdict | best_calibrated_judge | low_coverage_win | leakage_detected |
| ---: | --- | --- | ---: | ---: |
| 0 | SUPPORTED_LOW_COVERAGE | calibration_selected_candidate_ranker | True | False |
| 1 | SUPPORTED_LOW_COVERAGE | calibration_selected_candidate_ranker | True | False |
| 2 | SUPPORTED_LOW_COVERAGE | rank_normalized_linear | True | False |
| 3 | SUPPORTED_LOW_COVERAGE | calibration_selected_candidate_ranker | True | False |
| 4 | SUPPORTED_LOW_COVERAGE | rank_normalized_linear | True | False |
| 5 | SUPPORTED_LOW_COVERAGE | calibration_selected_candidate_ranker | True | False |
| 6 | SUPPORTED_LOW_COVERAGE | calibration_selected_candidate_ranker | True | False |
| 7 | SUPPORTED_LOW_COVERAGE | calibration_selected_candidate_ranker | True | False |
| 8 | SUPPORTED_LOW_COVERAGE | rank_normalized_linear | True | False |
| 9 | SUPPORTED_LOW_COVERAGE | calibration_selected_candidate_ranker | True | False |

## Low-coverage aggregate

| coverage | win_rate_vs_calibration_selected_single_signal | mean_margin | std_margin |
| ---: | ---: | ---: | ---: |
| 0.050000 | 1.000000 | 0.172000 | 0.019276 |
| 0.100000 | 1.000000 | 0.132667 | 0.015333 |

## Judge robustness

| judge | win_count | mean_far | std_far |
| --- | ---: | ---: | ---: |
| best_single_signal_selected_on_calibration | 300 | 0.649000 | 0.463048 |
| calibration_selected_candidate_ranker | 300 | 0.508000 | 0.484201 |
| combined_linear | 300 | 0.512667 | 0.473272 |
| conservative_low_coverage_judge | 300 | 0.565000 | 0.455301 |
| disagreement_only | 300 | 0.495333 | 0.484659 |
| invariant_only | 300 | 0.643333 | 0.460684 |
| isotonic_calibrated_judge | 300 | 0.646667 | 0.462242 |
| logistic_calibrated_judge | 300 | 0.545667 | 0.458237 |
| oracle_error_rank | 300 | 0.492333 | 0.486927 |
| quantile_rule_judge | 300 | 0.649000 | 0.463048 |
| random_baseline | 300 | 0.647667 | 0.462971 |
| rank_normalized_linear | 300 | 0.560000 | 0.464816 |
| repair_only | 300 | 0.648667 | 0.463047 |
| support_only | 300 | 0.648000 | 0.463983 |
| uncertainty_only | 300 | 0.655667 | 0.466684 |

## Failure cases

none

## Verdict

ROBUST_LOW_COVERAGE

## Explanation

Calibrated low-coverage wins appeared in 10 of 10 seeds.
