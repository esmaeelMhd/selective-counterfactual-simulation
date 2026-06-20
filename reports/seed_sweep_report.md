# Seed Sweep Report

## Command

```bash
python scripts/run_seed_sweep.py --config configs/experiments/smoke_two_tank.yaml --seeds 0 1 2 3 4 5 6 7 8 9 --output results/seed_sweep_two_tank
```

## Seeds

0, 1, 2, 3, 4, 5, 6, 7, 8, 9

## Per-seed verdict

| seed | verdict | overall_combined_win_rate | best_simple_judge | notes |
| ---: | --- | ---: | --- | --- |
| 0 | NOT_SUPPORTED | 0.000000 | support_only |  |
| 1 | NOT_SUPPORTED | 0.000000 | support_only |  |
| 2 | NOT_SUPPORTED | 0.000000 | support_only |  |
| 3 | NOT_SUPPORTED | 0.000000 | support_only |  |
| 4 | NOT_SUPPORTED | 0.000000 | support_only |  |
| 5 | NOT_SUPPORTED | 0.000000 | support_only |  |
| 6 | NOT_SUPPORTED | 0.000000 | support_only |  |
| 7 | NOT_SUPPORTED | 0.000000 | support_only |  |
| 8 | NOT_SUPPORTED | 0.000000 | support_only |  |
| 9 | NOT_SUPPORTED | 0.000000 | support_only |  |

## Aggregate result

| metric | mean | std | min | max |
| --- | ---: | ---: | ---: | ---: |
| overall_combined_win_rate | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## Combined judge robustness

Combined_linear reached the >= 0.70 per-seed win-rate threshold on 0 of 10 seeds.

## Split-level robustness

| split | combined_win_rate_mean | combined_win_rate_std |
| --- | ---: | ---: |
| id_test | 0.000000 | 0.000000 |
| ood_action_magnitude | 0.000000 | 0.000000 |
| ood_combined | 0.000000 | 0.000000 |
| ood_inflow_spike | 0.000000 | 0.000000 |

## Coverage-level robustness

| coverage | combined_win_rate_mean | combined_win_rate_std |
| ---: | ---: | ---: |
| 0.100000 | 0.000000 | 0.000000 |
| 0.200000 | 0.000000 | 0.000000 |
| 0.400000 | 0.000000 | 0.000000 |
| 0.600000 | 0.000000 | 0.000000 |
| 0.800000 | 0.000000 | 0.000000 |
| 1.000000 | 0.000000 | 0.000000 |

## Verdict

NOT_SUPPORTED

## Explanation

Average combined win rate across seeds was 0.000000; variance is reported above.

## Known failures

- none
