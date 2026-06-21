# Signal-Set Ablation: Full vs No Repair

## Question

Does excluding repair_amount improve CSTR without hurting TwoTank?

## Low-coverage result

| system | signal_set | coverage | baseline_far | calibrated_far | margin |
| --- | --- | ---: | ---: | ---: | ---: |
| two_tank | full_original | 0.050000 | 0.640000 | 0.466667 | 0.173333 |
| two_tank | full_original | 0.100000 | 0.653333 | 0.486667 | 0.166667 |
| two_tank | no_repair | 0.050000 | 0.640000 | 0.466667 | 0.173333 |
| two_tank | no_repair | 0.100000 | 0.653333 | 0.486667 | 0.166667 |
| two_tank | invariant_only | 0.050000 | 0.640000 | 0.626667 | 0.013333 |
| two_tank | invariant_only | 0.100000 | 0.653333 | 0.640000 | 0.013333 |
| two_tank | repair_only | 0.050000 | 0.666667 | 0.666667 | 0.000000 |
| two_tank | repair_only | 0.100000 | 0.653333 | 0.653333 | 0.000000 |
| two_tank | no_repair_no_uncertainty | 0.050000 | 0.640000 | 0.466667 | 0.173333 |
| two_tank | no_repair_no_uncertainty | 0.100000 | 0.653333 | 0.486667 | 0.166667 |
| cstr | full_original | 0.050000 | 0.666667 | 0.628571 | 0.038095 |
| cstr | full_original | 0.100000 | 0.666667 | 0.628571 | 0.038095 |
| cstr | no_repair | 0.050000 | 0.666667 | 0.628571 | 0.038095 |
| cstr | no_repair | 0.100000 | 0.666667 | 0.628571 | 0.038095 |
| cstr | invariant_only | 0.050000 | 0.666667 | 0.628571 | 0.038095 |
| cstr | invariant_only | 0.100000 | 0.666667 | 0.628571 | 0.038095 |
| cstr | repair_only | 0.050000 | 0.704762 | 0.704762 | 0.000000 |
| cstr | repair_only | 0.100000 | 0.728571 | 0.728571 | 0.000000 |
| cstr | no_repair_no_uncertainty | 0.050000 | 0.666667 | 0.628571 | 0.038095 |
| cstr | no_repair_no_uncertainty | 0.100000 | 0.666667 | 0.628571 | 0.038095 |

## Difference vs full_original

| system | signal_set | coverage | delta_margin_vs_full |
| --- | --- | ---: | ---: |
| two_tank | full_original | 0.050000 | 0.000000 |
| two_tank | full_original | 0.100000 | 0.000000 |
| two_tank | no_repair | 0.050000 | 0.000000 |
| two_tank | no_repair | 0.100000 | 0.000000 |
| two_tank | invariant_only | 0.050000 | -0.160000 |
| two_tank | invariant_only | 0.100000 | -0.153333 |
| two_tank | repair_only | 0.050000 | -0.173333 |
| two_tank | repair_only | 0.100000 | -0.166667 |
| two_tank | no_repair_no_uncertainty | 0.050000 | 0.000000 |
| two_tank | no_repair_no_uncertainty | 0.100000 | 0.000000 |
| cstr | full_original | 0.050000 | 0.000000 |
| cstr | full_original | 0.100000 | 0.000000 |
| cstr | no_repair | 0.050000 | 0.000000 |
| cstr | no_repair | 0.100000 | 0.000000 |
| cstr | invariant_only | 0.050000 | 0.000000 |
| cstr | invariant_only | 0.100000 | 0.000000 |
| cstr | repair_only | 0.050000 | -0.038095 |
| cstr | repair_only | 0.100000 | -0.038095 |
| cstr | no_repair_no_uncertainty | 0.050000 | 0.000000 |
| cstr | no_repair_no_uncertainty | 0.100000 | 0.000000 |

## Selected signals/judges

| system_id | signal_set_id | coverage | selected_judge | selected_signal_if_any |
| --- | --- | ---: | --- | --- |
| two_tank | full_original | 0.050000 | calibration_selected_candidate_ranker | none |
| two_tank | full_original | 0.100000 | rank_normalized_linear | none |
| two_tank | no_repair | 0.050000 | calibration_selected_candidate_ranker | none |
| two_tank | no_repair | 0.100000 | rank_normalized_linear | none |
| two_tank | invariant_only | 0.050000 | rank_normalized_linear | none |
| two_tank | invariant_only | 0.100000 | rank_normalized_linear | none |
| two_tank | repair_only | 0.050000 | rank_normalized_linear | none |
| two_tank | repair_only | 0.100000 | rank_normalized_linear | none |
| two_tank | no_repair_no_uncertainty | 0.050000 | calibration_selected_candidate_ranker | none |
| two_tank | no_repair_no_uncertainty | 0.100000 | rank_normalized_linear | none |
| cstr | full_original | 0.050000 | rank_normalized_linear | none |
| cstr | full_original | 0.100000 | rank_normalized_linear | none |
| cstr | no_repair | 0.050000 | rank_normalized_linear | none |
| cstr | no_repair | 0.100000 | rank_normalized_linear | none |
| cstr | invariant_only | 0.050000 | rank_normalized_linear | none |
| cstr | invariant_only | 0.100000 | rank_normalized_linear | none |
| cstr | repair_only | 0.050000 | rank_normalized_linear | none |
| cstr | repair_only | 0.100000 | rank_normalized_linear | none |
| cstr | no_repair_no_uncertainty | 0.050000 | rank_normalized_linear | none |
| cstr | no_repair_no_uncertainty | 0.100000 | rank_normalized_linear | none |

## Leakage status

| system_id | signal_set_id | leakage_detected |
| --- | --- | ---: |
| cstr | full_original | False |
| cstr | invariant_only | False |
| cstr | no_repair | False |
| cstr | no_repair_no_uncertainty | False |
| cstr | repair_only | False |
| two_tank | full_original | False |
| two_tank | invariant_only | False |
| two_tank | no_repair | False |
| two_tank | no_repair_no_uncertainty | False |
| two_tank | repair_only | False |

## Interpretation

CSTR no-repair delta=0.000000; TwoTank no-repair delta=0.000000.

## Verdict

NO_REPAIR_NO_BENEFIT
