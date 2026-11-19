# Exploratory BSM1 model check

Scored on `judge_calibration_id` after fitting `model_train`.
This is not the eligibility rule. Eligibility uses `judge_test_id`, and that split is not generated.

| model | fit seconds | state RMSE | one-step RMSE | flow | temperature | settler | concentration | effluent RMSE | final train loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hold_last | 0.0 | 1429 | 419.8 | 6789 | 0 | 320.4 | 35.2 | 1.596 | nan |
| linear_narx | 0.3 | 13.65 | 2.526 | 1.467e-05 | 0 | 49.42 | 2.845 | 0.2622 | nan |
| lstm_world | 7.6 | 121.9 | 43.8 | 486.3 | 6.491e-10 | 233.3 | 22.38 | 1.283 | 0.0006107 |
| recurrent_world | 16.0 | 176 | 82.79 | 780.3 | 0 | 224 | 22.86 | 1.685 | 0.004584 |
| mlp_state_space | 4.5 | 406.1 | 64.16 | 1898 | 1.56e-05 | 282.7 | 14.01 | 1.033 | nan |
