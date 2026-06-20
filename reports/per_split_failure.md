# Per-Split Failure Analysis

## Worst splits by model error

| split | scenario_type | mean_rmse | bad_rmse_rate |
| --- | --- | ---: | ---: |
| ood_combined | combined_intervention | 0.474653 | 1.000000 |
| ood_action_magnitude | held_out_action_magnitude | 0.420843 | 0.966667 |
| ood_inflow_spike | inflow_spike | 0.270846 | 0.666667 |
| id_test | normal_policy | 0.171247 | 0.300000 |

## Worst splits by false accepts

| split | scenario_type | judge | coverage | false_accept_rate |
| --- | --- | --- | ---: | ---: |
| ood_inflow_spike | inflow_spike | support_only | 1.000000 | 1.000000 |
| ood_inflow_spike | inflow_spike | uncertainty_only | 1.000000 | 1.000000 |
| ood_action_magnitude | held_out_action_magnitude | repair_only | 1.000000 | 1.000000 |
| ood_action_magnitude | held_out_action_magnitude | support_only | 1.000000 | 1.000000 |
| ood_action_magnitude | held_out_action_magnitude | uncertainty_only | 1.000000 | 1.000000 |
| ood_action_magnitude | held_out_action_magnitude | invariant_only | 1.000000 | 1.000000 |
| ood_action_magnitude | held_out_action_magnitude | oracle_error_rank | 1.000000 | 1.000000 |
| ood_action_magnitude | held_out_action_magnitude | random_baseline | 1.000000 | 1.000000 |
| ood_action_magnitude | held_out_action_magnitude | combined_linear | 1.000000 | 1.000000 |
| ood_action_magnitude | held_out_action_magnitude | disagreement_only | 1.000000 | 1.000000 |
| ood_combined | combined_intervention | combined_linear | 1.000000 | 1.000000 |
| ood_combined | combined_intervention | disagreement_only | 1.000000 | 1.000000 |

## Best real judge per split

| split | best_real_judge | false_accept_rate | combined_linear_far | combined_margin |
| --- | --- | ---: | ---: | ---: |
| id_test | disagreement_only | 0.138889 | 0.140000 | -0.001111 |
| ood_action_magnitude | disagreement_only | 0.484444 | 0.486667 | -0.002222 |
| ood_combined | disagreement_only | 0.516667 | 0.516667 | 0.000000 |
| ood_inflow_spike | disagreement_only | 0.344444 | 0.344444 | 0.000000 |

## Where combined_linear failed

ood_action_magnitude, id_test, ood_combined, ood_inflow_spike

## Where combined_linear worked

none

## Interpretation

Worst split by mean RMSE: ood_combined; combined failures are listed above.

## Verdict

GLOBAL_FAILURE
