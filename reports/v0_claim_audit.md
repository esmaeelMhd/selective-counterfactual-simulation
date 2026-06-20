# v0 Claim Audit

## Main question

Did combined_linear reduce false_accept_rate compared with the strongest simple judge?

## Data sources

- results/smoke_two_tank/risk_coverage.csv
- results/smoke_two_tank/scenario_scores.csv
- results/smoke_two_tank/model_metrics.csv
- results/smoke_two_tank/summary.json

## Judge definitions

Real judges: combined_linear, support_only, uncertainty_only, disagreement_only, invariant_only, repair_only, random_baseline.
Diagnostic judge excluded from real-method ranking: oracle_error_rank.

## Result by OOD split

| split | coverage | best_simple_judge | best_simple_far | combined_far | combined_margin | combined_wins |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| ood_action_magnitude | 0.100000 | disagreement_only | 0.200000 | 0.400000 | -0.200000 | 0.000000 |
| ood_action_magnitude | 0.100000 | support_only | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| ood_action_magnitude | 0.200000 | disagreement_only | 0.500000 | 0.600000 | -0.100000 | 0.000000 |
| ood_action_magnitude | 0.200000 | support_only | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| ood_action_magnitude | 0.400000 | disagreement_only | 0.750000 | 0.750000 | 0.000000 | 0.000000 |
| ood_action_magnitude | 0.400000 | support_only | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| ood_action_magnitude | 0.600000 | disagreement_only | 0.833333 | 0.833333 | 0.000000 | 0.000000 |
| ood_action_magnitude | 0.600000 | support_only | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| ood_action_magnitude | 0.800000 | disagreement_only | 0.875000 | 0.875000 | 0.000000 | 0.000000 |
| ood_action_magnitude | 0.800000 | support_only | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| ood_action_magnitude | 1.000000 | support_only | 0.966667 | 0.966667 | 0.000000 | 0.000000 |
| ood_combined | 0.100000 | support_only | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| ood_combined | 0.200000 | support_only | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| ood_combined | 0.400000 | support_only | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| ood_combined | 0.600000 | support_only | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| ood_combined | 0.800000 | support_only | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| ood_combined | 1.000000 | support_only | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| ood_inflow_spike | 0.100000 | support_only | 0.666667 | 0.666667 | 0.000000 | 0.000000 |
| ood_inflow_spike | 0.200000 | support_only | 0.666667 | 0.666667 | 0.000000 | 0.000000 |
| ood_inflow_spike | 0.400000 | support_only | 0.666667 | 0.666667 | 0.000000 | 0.000000 |
| ood_inflow_spike | 0.600000 | support_only | 0.666667 | 0.666667 | 0.000000 | 0.000000 |
| ood_inflow_spike | 0.800000 | support_only | 0.666667 | 0.666667 | 0.000000 | 0.000000 |
| ood_inflow_spike | 1.000000 | support_only | 0.666667 | 0.666667 | 0.000000 | 0.000000 |

## Result by model

| model_id | best_simple_judge | combined_win_rate | verdict |
| --- | --- | ---: | --- |
| hold_last | support_only | 0.000000 | NOT_SUPPORTED |
| linear_narx | support_only | 0.000000 | NOT_SUPPORTED |
| mlp_state_space | support_only | 0.000000 | NOT_SUPPORTED |

## Result by coverage

| coverage | combined_win_rate |
| ---: | ---: |
| 0.100000 | 0.000000 |
| 0.200000 | 0.000000 |
| 0.400000 | 0.000000 |
| 0.600000 | 0.000000 |
| 0.800000 | 0.000000 |
| 1.000000 | 0.000000 |

## Oracle gap

Mean oracle gap: 0.002778; min: 0.000000; max: 0.200000.

## Verdict

NOT_SUPPORTED

## Explanation

Overall strict win rate was 0.000000. Combined_linear won on 0 OOD splits under the split-level rule. Verdict: NOT_SUPPORTED.

## Known failure modes

- Ties are not counted as wins.
- Oracle_error_rank is diagnostic only and excluded from strongest-simple comparisons.
