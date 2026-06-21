# Per-Split Failure Analysis

## Worst splits by model error

| split | scenario_type | mean_rmse | bad_rmse_rate |
| --- | --- | ---: | ---: |
| ood_combined | combined_intervention | 0.242500 | 1.000000 |
| ood_action_magnitude | held_out_action_magnitude | 0.190000 | 0.750000 |
| ood_inflow_spike | inflow_spike | 0.162500 | 0.583333 |
| id_test | normal_policy | 0.087500 | 0.000000 |

## Worst splits by false accepts

| split | scenario_type | judge | coverage | false_accept_rate |
| --- | --- | --- | ---: | ---: |
| ood_combined | combined_intervention | invariant_only | 1.000000 | 1.000000 |
| ood_combined | combined_intervention | oracle_error_rank | 1.000000 | 1.000000 |
| ood_combined | combined_intervention | random_baseline | 1.000000 | 1.000000 |
| ood_combined | combined_intervention | repair_only | 1.000000 | 1.000000 |
| ood_combined | combined_intervention | support_only | 1.000000 | 1.000000 |
| ood_combined | combined_intervention | uncertainty_only | 1.000000 | 1.000000 |
| ood_combined | combined_intervention | combined_linear | 1.000000 | 1.000000 |
| ood_combined | combined_intervention | disagreement_only | 1.000000 | 1.000000 |
| ood_combined | combined_intervention | combined_linear | 1.000000 | 1.000000 |
| ood_combined | combined_intervention | disagreement_only | 1.000000 | 1.000000 |
| ood_combined | combined_intervention | oracle_error_rank | 1.000000 | 1.000000 |
| ood_combined | combined_intervention | invariant_only | 1.000000 | 1.000000 |

## Best real judge per split

| split | best_real_judge | false_accept_rate | combined_linear_far | combined_margin |
| --- | --- | ---: | ---: | ---: |
| id_test | disagreement_only | 0.000000 | 0.000000 | 0.000000 |
| ood_action_magnitude | disagreement_only | 0.361111 | 0.361111 | 0.000000 |
| ood_combined | disagreement_only | 0.611111 | 0.611111 | 0.000000 |
| ood_inflow_spike | disagreement_only | 0.222222 | 0.222222 | 0.000000 |

## Where combined_linear failed

id_test, ood_action_magnitude, ood_combined, ood_inflow_spike

## Where combined_linear worked

none

## Interpretation

Worst split by mean RMSE: ood_combined; combined failures are listed above.

## Verdict

GLOBAL_FAILURE
