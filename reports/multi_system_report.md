# Multi-System Report

## Systems

cstr, two_tank

## Models

hold_last, linear_narx, mlp_state_space

## Judges

combined_linear, disagreement_only, invariant_only, oracle_error_rank, random_baseline, repair_only, support_only, uncertainty_only

## Claim status by system

| system | claim_verdict | seed_verdict | severity_verdict |
| --- | --- | --- | --- |
| two_tank | NOT_SUPPORTED | NOT_SUPPORTED | MEANINGFUL |
| cstr | NOT_SUPPORTED | NOT_SUPPORTED | NOT_RUN |

## Combined judge vs strongest simple judge

| system | combined_win_rate | best_simple_judge |
| --- | ---: | --- |
| two_tank | 0.000000 | support_only |
| cstr | 0.000000 | support_only |

## Overall claim status

NOT_SUPPORTED

## Explanation

The v0 decision gate is KILL_OR_DOWNGRADE_CLAIM, so expansion outputs are reported as diagnostics only.

## Known failures

- Decision gate is KILL_OR_DOWNGRADE_CLAIM; expansion is diagnostic, not claim support.
- At least one system has claim_verdict=NOT_SUPPORTED.
