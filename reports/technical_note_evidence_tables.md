# Technical Note Evidence Tables

## Main low-coverage result

| system | coverage | baseline_far | calibrated_far | margin | effect_strength |
| --- | ---: | ---: | ---: | ---: | --- |
| two_tank | 0.050000 | 0.640000 | 0.466667 | 0.173333 | practically_meaningful |
| two_tank | 0.100000 | 0.653333 | 0.486667 | 0.166667 | practically_meaningful |
| cstr | 0.050000 | 0.666667 | 0.628571 | 0.038095 | positive_but_weak |
| cstr | 0.100000 | 0.666667 | 0.628571 | 0.038095 | positive_but_weak |

## Smoke model sanity

| model | id_rmse_mean |
| --- | ---: |
| hold_last | 0.504806 |
| linear_narx | 0.007754 |
| mlp_state_space | 0.001180 |

## Signal semantics

| signal | system | role | key_finding |
| --- | --- | --- | --- |
| repair_amount | cstr | diagnostic_only | AUROC 0.500000; diagnostic-only for within-bound CSTR errors |
| invariant_residual | cstr | informative_refusal_signal | AUROC 0.954061; much more informative for CSTR |

## Claim status

| claim | status | evidence |
| --- | --- | --- |
| combined_linear works | not supported as the original broad claim | v0 decision gate killed or downgraded the original combined_linear claim |
| calibrated low-coverage works on TwoTank | supported | TwoTank low-coverage margins are practically meaningful |
| calibrated low-coverage weakly replicates on CSTR | weak positive | CSTR margins are positive but small |
| repair_amount is universal | false | CSTR repair AUROC is 0.5 and role gate marks repair diagnostic-only |
| invariant_residual is informative on CSTR | supported | CSTR invariant residual AUROC is high in the repair-vs-invariant audit |
| general simulator reliability | forbidden | Current status gate blocks expansion and general reliability claims |
| product readiness | forbidden | The repo has no product layer and the current status forbids product claims |
