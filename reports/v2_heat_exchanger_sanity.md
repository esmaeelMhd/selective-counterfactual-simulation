# v2 Heat Exchanger Sanity

## Verdict

VALID_HEAT_EXCHANGER_BENCHMARK

## Checks

- finite trajectories: True
- nonconstant trajectories: True
- OOD differs from ID: True
- OOD linear RMSE > ID linear RMSE: True
- bad RMSE labels non-degenerate: True
- event labels non-degenerate: True
- calibration/test overlap count: 0

## Distribution Checks

| split | scenario_type | action_range | hot_inlet_range | ood_differs_from_id |
| --- | --- | ---: | ---: | ---: |
| judge_calibration_id | id | 0.511814 | 7.857743 | False |
| judge_calibration_inlet_temperature_spike | inlet_temperature_spike | 0.489501 | 48.854854 | False |
| judge_calibration_flow_rate_shift | flow_rate_shift | 1.353467 | 7.433518 | False |
| judge_calibration_cooling_or_heating_change | cooling_or_heating_change | 1.680805 | 7.345582 | False |
| judge_calibration_heat_transfer_coefficient_shift | heat_transfer_coefficient_shift | 1.055013 | 21.341108 | False |
| judge_calibration_combined_disturbance_shift | combined_disturbance_shift | 1.244758 | 38.712180 | False |
| judge_calibration_unsafe_outlet_temperature_event | unsafe_outlet_temperature_event | 1.190199 | 63.456509 | False |
| judge_test_id | id | 0.518988 | 7.668204 | False |
| judge_test_inlet_temperature_spike | inlet_temperature_spike | 0.492041 | 49.230013 | True |
| judge_test_flow_rate_shift | flow_rate_shift | 1.432204 | 7.182956 | True |
| judge_test_cooling_or_heating_change | cooling_or_heating_change | 1.642635 | 8.008508 | True |
| judge_test_heat_transfer_coefficient_shift | heat_transfer_coefficient_shift | 1.030273 | 20.207396 | True |

## Model Error Checks

| model_id | split | rmse_mean |
| --- | --- | ---: |
| hold_last | judge_calibration_id | 4.187189 |
| hold_last | judge_calibration_inlet_temperature_spike | 6.248291 |
| hold_last | judge_calibration_flow_rate_shift | 3.400269 |
| hold_last | judge_calibration_cooling_or_heating_change | 3.774518 |
| hold_last | judge_calibration_heat_transfer_coefficient_shift | 5.756058 |
| hold_last | judge_calibration_combined_disturbance_shift | 6.411421 |
| hold_last | judge_calibration_unsafe_outlet_temperature_event | 12.036673 |
| hold_last | judge_test_id | 4.096865 |
| hold_last | judge_test_inlet_temperature_spike | 6.743777 |
| hold_last | judge_test_flow_rate_shift | 3.252785 |
| hold_last | judge_test_cooling_or_heating_change | 4.130957 |
| hold_last | judge_test_heat_transfer_coefficient_shift | 5.687388 |
