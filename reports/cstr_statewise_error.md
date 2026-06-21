# CSTR State-Wise Error Decomposition

## Question

Is CSTR weakness dominated by concentration error, temperature error, or both?

## State-wise error by model

| model | concentration_rmse | temperature_rmse | dominant_state |
| --- | ---: | ---: | --- |
| hold_last | 0.194456 | 6.274258 | concentration |
| linear_narx | 0.004536 | 0.352870 | concentration |
| mlp_state_space | 0.031341 | 2.485710 | concentration |

## State-wise error by scenario type

| scenario_type | concentration_rmse | temperature_rmse | dominant_state |
| --- | ---: | ---: | --- |
| combined_feed_and_cooling_shift | 0.072887 | 3.383326 | concentration |
| cooling_step_change | 0.082779 | 2.325633 | concentration |
| feed_concentration_spike | 0.076150 | 1.821497 | concentration |
| feed_temperature_spike | 0.071807 | 1.647042 | concentration |
| id | 0.074233 | 1.327081 | concentration |
| reaction_rate_shift | 0.083965 | 1.520613 | concentration |
| unsafe_temperature_event | 0.075625 | 9.238098 | concentration |

## Accepted false accepts

| coverage | model | scenario_type | concentration_rmse | temperature_rmse | dominant_state |
| ---: | --- | --- | ---: | ---: | --- |
| 0.050000 | hold_last | combined_feed_and_cooling_shift | 0.038355 | 6.842519 | temperature |
| 0.050000 | hold_last | cooling_step_change | 0.169131 | 3.579732 | concentration |
| 0.050000 | hold_last | feed_concentration_spike | 0.058246 | 3.459361 | concentration |
| 0.050000 | hold_last | feed_temperature_spike | 0.068356 | 3.165265 | concentration |
| 0.050000 | hold_last | id | 0.074914 | 2.374255 | concentration |
| 0.050000 | hold_last | reaction_rate_shift | 0.053516 | 2.195526 | concentration |
| 0.050000 | hold_last | unsafe_temperature_event | 0.039980 | 14.648583 | temperature |
| 0.050000 | linear_narx | unsafe_temperature_event | 0.006759 | 1.425918 | temperature |
| 0.050000 | mlp_state_space | combined_feed_and_cooling_shift | 0.052931 | 1.936224 | concentration |
| 0.050000 | mlp_state_space | cooling_step_change | 0.024282 | 2.986248 | concentration |
| 0.050000 | mlp_state_space | feed_concentration_spike | 0.042211 | 0.553450 | concentration |
| 0.050000 | mlp_state_space | feed_temperature_spike | 0.006498 | 0.320905 | concentration |
| 0.050000 | mlp_state_space | reaction_rate_shift | 0.054569 | 0.307915 | concentration |
| 0.050000 | mlp_state_space | unsafe_temperature_event | 0.016611 | 10.030817 | temperature |
| 0.100000 | hold_last | combined_feed_and_cooling_shift | 0.038483 | 6.905631 | temperature |
| 0.100000 | hold_last | cooling_step_change | 0.175613 | 3.547478 | concentration |
| 0.100000 | hold_last | feed_concentration_spike | 0.061729 | 3.590450 | concentration |
| 0.100000 | hold_last | feed_temperature_spike | 0.067019 | 3.331019 | concentration |
| 0.100000 | hold_last | id | 0.081051 | 2.732042 | concentration |
| 0.100000 | hold_last | reaction_rate_shift | 0.039564 | 2.685047 | concentration |
| 0.100000 | hold_last | unsafe_temperature_event | 0.050129 | 14.665611 | temperature |
| 0.100000 | linear_narx | unsafe_temperature_event | 0.007084 | 1.442237 | temperature |
| 0.100000 | mlp_state_space | combined_feed_and_cooling_shift | 0.051726 | 1.959749 | concentration |
| 0.100000 | mlp_state_space | cooling_step_change | 0.023345 | 2.993760 | temperature |
| 0.100000 | mlp_state_space | feed_concentration_spike | 0.042142 | 0.560229 | concentration |
| 0.100000 | mlp_state_space | feed_temperature_spike | 0.007283 | 0.337493 | concentration |
| 0.100000 | mlp_state_space | reaction_rate_shift | 0.053450 | 0.344824 | concentration |
| 0.100000 | mlp_state_space | unsafe_temperature_event | 0.015644 | 10.071814 | temperature |

## Verdict

BOTH_STATES
