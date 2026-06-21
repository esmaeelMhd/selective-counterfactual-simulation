# Repair Amount Semantics Validation

## Question

Is repair_amount broken or semantically irrelevant for CSTR?

## Controlled cases

| system | case | violates_bounds | repair_amount | expected | case_passed |
| --- | --- | ---: | ---: | --- | ---: |
| two_tank | valid_in_bounds_trajectory | False | 0.000000 | near_zero | True |
| two_tank | out_of_bounds_negative_inventory | True | 0.048077 | positive | True |
| two_tank | out_of_bounds_over_capacity | True | 0.076923 | positive | True |
| two_tank | within_bounds_wrong_dynamics | False | 0.000000 | near_zero | True |
| cstr | valid_in_bounds_trajectory | False | 0.000000 | near_zero | True |
| cstr | out_of_bounds_temperature | True | 1.153846 | positive | True |
| cstr | out_of_bounds_concentration | True | 0.015385 | positive | True |
| cstr | within_bounds_wrong_reaction_dynamics | False | 0.000000 | near_zero | True |
| cstr | within_bounds_wrong_temperature_trajectory | False | 0.000000 | near_zero | True |

## CSTR repair status

REPAIR_BOUNDS_ONLY

## TwoTank repair status

REPAIR_BOUNDS_ONLY

## Interpretation

within-bound CSTR errors are high-RMSE but near-zero repair

## Verdict

REPAIR_CORRECT_BUT_CSTR_IRRELEVANT
