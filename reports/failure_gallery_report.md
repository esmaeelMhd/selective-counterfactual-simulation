# Failure Gallery Report

## Source artifacts

results/calibrated_two_tank/test_table.csv, results/calibrated_cstr/test_table.csv

## Examples

| example_id | system | model | decision | false_accept | rmse | repair_amount | invariant_residual |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| example_1_accepted_good | two_tank | mlp_state_space | accepted | False | 0.000451 | 0.000000 | 0.000101 |
| example_2_correctly_rejected_bad | two_tank | hold_last | refused | False | 1.380867 | 0.000000 | 0.005934 |
| example_3_false_accept_cstr | cstr | hold_last | accepted | True | 3.785052 | 0.000000 | 0.359716 |
| example_4_cstr_within_bound_dynamic_failure | cstr | hold_last | refused | False | 12.118251 | 0.000000 | 0.810614 |
| example_5_invariant_residual_helps | cstr | hold_last | refused | False | 11.961385 | 0.000000 | 0.850472 |

## Verdict

FAILURE_GALLERY_BUILT
