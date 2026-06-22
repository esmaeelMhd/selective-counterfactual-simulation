# Smoke Demo Report

This smoke demo checks that the benchmark pipeline runs; it is not the full evidence reproduction.

## Scope

System: two_tank

Model: linear_narx

## Split RMSE

| split | mean_rmse | n_scenarios |
| --- | ---: | ---: |
| id_test | 0.006369 | 6.000000 |
| ood_action_magnitude | 0.210734 | 6.000000 |
| ood_inflow_spike | 0.008912 | 6.000000 |
| ood_combined | 0.112319 | 6.000000 |

## Claim boundary

This smoke output is not evidence for the current supported claim.

## Verdict

SMOKE_DEMO_BUILT
