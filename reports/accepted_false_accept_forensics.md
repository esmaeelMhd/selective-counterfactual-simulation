# Accepted False-Accept Forensics

## Question

Which bad scenarios are still accepted by the calibrated judge?

## False accepts by system

| system | coverage | accepted_count | accepted_bad_count | accepted_bad_rate |
| --- | ---: | ---: | ---: | ---: |
| two_tank | 0.050000 | 75 | 35 | 0.466667 |
| two_tank | 0.100000 | 150 | 73 | 0.486667 |
| cstr | 0.050000 | 105 | 66 | 0.628571 |
| cstr | 0.100000 | 210 | 132 | 0.628571 |

## False accepts by model

| system | model | accepted_bad_count | mean_bad_rmse |
| --- | --- | ---: | ---: |
| cstr | hold_last | 35 | 3.664111 |
| cstr | hold_last | 70 | 3.784536 |
| cstr | linear_narx | 5 | 1.008288 |
| cstr | linear_narx | 10 | 1.019828 |
| cstr | mlp_state_space | 26 | 2.161123 |
| cstr | mlp_state_space | 50 | 2.252206 |
| cstr | mlp_state_space | 2 | 0.246786 |
| two_tank | hold_last | 10 | 0.411914 |
| two_tank | hold_last | 23 | 0.381821 |
| two_tank | linear_narx | 10 | 0.389464 |
| two_tank | linear_narx | 20 | 0.387592 |
| two_tank | mlp_state_space | 15 | 0.257322 |
| two_tank | mlp_state_space | 30 | 0.258149 |

## False accepts by scenario type

| system | scenario_type | accepted_bad_count | mean_bad_rmse |
| --- | --- | ---: | ---: |
| cstr | combined_feed_and_cooling_shift | 5 | 4.838468 |
| cstr | combined_feed_and_cooling_shift | 10 | 4.883095 |
| cstr | cooling_step_change | 5 | 2.534127 |
| cstr | cooling_step_change | 10 | 2.511604 |
| cstr | feed_concentration_spike | 5 | 2.446490 |
| cstr | feed_concentration_spike | 10 | 2.539225 |
| cstr | feed_temperature_spike | 5 | 2.238762 |
| cstr | feed_temperature_spike | 10 | 2.355935 |
| cstr | id | 5 | 1.679784 |
| cstr | id | 10 | 1.932759 |
| cstr | reaction_rate_shift | 5 | 1.552990 |
| cstr | reaction_rate_shift | 10 | 1.898912 |
| cstr | unsafe_temperature_event | 5 | 10.358157 |
| cstr | unsafe_temperature_event | 10 | 10.370224 |
| cstr | unsafe_temperature_event | 5 | 1.008288 |
| cstr | unsafe_temperature_event | 10 | 1.019828 |
| cstr | combined_feed_and_cooling_shift | 5 | 1.369631 |
| cstr | combined_feed_and_cooling_shift | 10 | 1.386237 |
| cstr | cooling_step_change | 5 | 2.111666 |
| cstr | cooling_step_change | 10 | 2.116973 |
| cstr | feed_concentration_spike | 5 | 0.392487 |
| cstr | feed_concentration_spike | 10 | 0.397263 |
| cstr | feed_temperature_spike | 5 | 0.226962 |
| cstr | feed_temperature_spike | 10 | 0.238701 |
| cstr | reaction_rate_shift | 1 | 0.221122 |
| cstr | reaction_rate_shift | 2 | 0.246786 |
| cstr | unsafe_temperature_event | 5 | 7.092868 |
| cstr | unsafe_temperature_event | 10 | 7.121857 |
| two_tank | combined_intervention | 5 | 0.448073 |
| two_tank | combined_intervention | 10 | 0.453085 |
| two_tank | held_out_action_magnitude | 1 | 0.158399 |
| two_tank | inflow_spike | 5 | 0.375754 |
| two_tank | inflow_spike | 10 | 0.378093 |
| two_tank | normal_policy | 1 | 0.154500 |
| two_tank | valve_or_pump_degradation | 1 | 0.157190 |
| two_tank | combined_intervention | 5 | 0.365399 |
| two_tank | combined_intervention | 10 | 0.362634 |
| two_tank | held_out_action_magnitude | 5 | 0.413528 |
| two_tank | held_out_action_magnitude | 10 | 0.412549 |
| two_tank | combined_intervention | 5 | 0.329693 |

## Diagnostic tag counts

| tag | count |
| --- | ---: |
| LOW_REPAIR_BUT_BAD | 306 |
| SEVERE_MISCLASSIFICATION | 171 |
| MODEL_SPECIFIC_FAILURE | 150 |
| LOW_DISAGREEMENT_BUT_BAD | 126 |
| LOW_UNCERTAINTY_BUT_BAD | 119 |
| LOW_INVARIANT_RESIDUAL_BUT_BAD | 45 |
| SPLIT_SPECIFIC_FAILURE | 45 |
| LOW_SUPPORT_RISK_BUT_BAD | 36 |
| NEAR_THRESHOLD_FAILURE | 18 |

## Worst accepted false accepts

| system | scenario_id | model | scenario_type | rmse | tags |
| --- | --- | --- | --- | ---: | --- |
| cstr | judge_test_unsafe_temperature_event_0010 | hold_last | unsafe_temperature_event | 10.727371 | LOW_UNCERTAINTY_BUT_BAD;LOW_REPAIR_BUT_BAD;MODEL_SPECIFIC_FAILURE;SEVERE_MISCLASSIFICATION |
| cstr | judge_test_unsafe_temperature_event_0010 | hold_last | unsafe_temperature_event | 10.727371 | LOW_UNCERTAINTY_BUT_BAD;LOW_REPAIR_BUT_BAD;MODEL_SPECIFIC_FAILURE;SEVERE_MISCLASSIFICATION |
| cstr | judge_test_unsafe_temperature_event_0023 | hold_last | unsafe_temperature_event | 10.659669 | LOW_UNCERTAINTY_BUT_BAD;LOW_REPAIR_BUT_BAD;MODEL_SPECIFIC_FAILURE;SEVERE_MISCLASSIFICATION |
| cstr | judge_test_unsafe_temperature_event_0031 | hold_last | unsafe_temperature_event | 10.639951 | LOW_UNCERTAINTY_BUT_BAD;LOW_REPAIR_BUT_BAD;MODEL_SPECIFIC_FAILURE;SEVERE_MISCLASSIFICATION |
| cstr | judge_test_unsafe_temperature_event_0004 | hold_last | unsafe_temperature_event | 10.485849 | LOW_REPAIR_BUT_BAD;MODEL_SPECIFIC_FAILURE;SEVERE_MISCLASSIFICATION |
| cstr | judge_test_unsafe_temperature_event_0004 | hold_last | unsafe_temperature_event | 10.485849 | LOW_REPAIR_BUT_BAD;MODEL_SPECIFIC_FAILURE;SEVERE_MISCLASSIFICATION |
| cstr | judge_test_unsafe_temperature_event_0007 | hold_last | unsafe_temperature_event | 10.337740 | LOW_UNCERTAINTY_BUT_BAD;LOW_REPAIR_BUT_BAD;MODEL_SPECIFIC_FAILURE;SEVERE_MISCLASSIFICATION |
| cstr | judge_test_unsafe_temperature_event_0007 | hold_last | unsafe_temperature_event | 10.337740 | LOW_UNCERTAINTY_BUT_BAD;LOW_REPAIR_BUT_BAD;MODEL_SPECIFIC_FAILURE;SEVERE_MISCLASSIFICATION |
| cstr | judge_test_unsafe_temperature_event_0076 | hold_last | unsafe_temperature_event | 10.273621 | LOW_UNCERTAINTY_BUT_BAD;LOW_REPAIR_BUT_BAD;MODEL_SPECIFIC_FAILURE;SEVERE_MISCLASSIFICATION |
| cstr | judge_test_unsafe_temperature_event_0046 | hold_last | unsafe_temperature_event | 10.217831 | LOW_UNCERTAINTY_BUT_BAD;LOW_REPAIR_BUT_BAD;MODEL_SPECIFIC_FAILURE;SEVERE_MISCLASSIFICATION |
| cstr | judge_test_unsafe_temperature_event_0046 | hold_last | unsafe_temperature_event | 10.217831 | LOW_UNCERTAINTY_BUT_BAD;LOW_REPAIR_BUT_BAD;MODEL_SPECIFIC_FAILURE;SEVERE_MISCLASSIFICATION |
| cstr | judge_test_unsafe_temperature_event_0093 | hold_last | unsafe_temperature_event | 10.191677 | LOW_UNCERTAINTY_BUT_BAD;LOW_REPAIR_BUT_BAD;MODEL_SPECIFIC_FAILURE;SEVERE_MISCLASSIFICATION |

## Interpretation

The tags are diagnostic only and are defined in `docs/false_accept_forensics_tags.md`.

## Verdict

INCONCLUSIVE

## Recommended next action

Inspect the dominant model/scenario clusters before changing any refusal signal.
