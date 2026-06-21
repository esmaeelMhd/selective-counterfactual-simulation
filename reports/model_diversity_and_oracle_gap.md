# Model Diversity and Oracle Gap Report

## Model error ranking

| split | best_model | worst_model | error_gap |
| --- | --- | --- | ---: |
| id_test | hold_last | linear_narx | 0.015000 |
| ood_action_magnitude | hold_last | linear_narx | 0.015000 |
| ood_inflow_spike | hold_last | linear_narx | 0.015000 |
| ood_combined | hold_last | linear_narx | 0.015000 |

## Model disagreement

| split | mean_pairwise_disagreement |
| --- | ---: |
| id_test | 0.052500 |
| ood_action_magnitude | 0.172500 |
| ood_inflow_spike | 0.292500 |
| ood_combined | 0.412500 |

## Oracle gap

| split | oracle_far | best_real_judge_far | oracle_gap |
| --- | ---: | ---: | ---: |
| id_test | 0.000000 | 0.000000 | 0.000000 |
| ood_action_magnitude | 0.361111 | 0.361111 | 0.000000 |
| ood_inflow_spike | 0.222222 | 0.222222 | 0.000000 |
| ood_combined | 0.611111 | 0.611111 | 0.000000 |

## Interpretation

Mean oracle gap was 0.000000. Oracle is diagnostic and not deployable.

## Verdict

ORACLE_GAP_SMALL
