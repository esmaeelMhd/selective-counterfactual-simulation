# Exploratory BSM1 model check

Scored on `judge_calibration_id` after fitting `model_train`.
This is not the eligibility rule. Eligibility uses `judge_test_id`, and that split is not generated.
Reactor and effluent flow are set from the influent with the open-loop recycle and waste flows.

| model | fit seconds | state RMSE | one-step RMSE | flow | temperature | settler | concentration | effluent RMSE | final train loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hold_last | 0.0 | 92.66 | 3.112 | 4.116e-13 | 0 | 320.4 | 35.2 | 1.596 | nan |
| linear_narx | 0.3 | 13.65 | 2.526 | 4.116e-13 | 0 | 49.42 | 2.845 | 0.2622 | nan |
| lstm_world | 7.9 | 71.95 | 3.475 | 4.116e-13 | 6.853e-10 | 252.2 | 24.45 | 1.339 | 0.0006107 |
| recurrent_world | 65.0 | 62.39 | 11.21 | 4.116e-13 | 0 | 217.8 | 22.02 | 1.634 | 0.004584 |
| mlp_state_space | 13.6 | 99.27 | 2.317 | 4.116e-13 | 1.584e-05 | 363 | 14.04 | 1.245 | nan |
