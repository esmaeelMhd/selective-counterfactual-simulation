# Model Diversity and Oracle Gap Report

## Model error ranking

| split | best_model | worst_model | error_gap |
| --- | --- | --- | ---: |
| id_test | mlp_state_space | hold_last | 0.503626 |
| ood_action_magnitude | mlp_state_space | hold_last | 0.346405 |
| ood_combined | mlp_state_space | hold_last | 0.345307 |
| ood_inflow_spike | linear_narx | hold_last | 0.557674 |

## Model disagreement

| split | mean_pairwise_disagreement |
| --- | ---: |
| id_test | 0.338624 |
| ood_action_magnitude | 0.661497 |
| ood_combined | 0.674878 |
| ood_inflow_spike | 0.425460 |

## Oracle gap

| split | oracle_far | best_real_judge_far | oracle_gap |
| --- | ---: | ---: | ---: |
| id_test | 0.138889 | 0.138889 | 0.000000 |
| ood_action_magnitude | 0.483333 | 0.484444 | 0.001111 |
| ood_combined | 0.516667 | 0.516667 | 0.000000 |
| ood_inflow_spike | 0.344444 | 0.344444 | 0.000000 |

## Interpretation

Mean oracle gap was 0.000278. Oracle is diagnostic and not deployable.

## Verdict

ORACLE_GAP_SMALL
