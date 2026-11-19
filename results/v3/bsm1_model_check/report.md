# Exploratory BSM1 model check

Scored on `judge_calibration_id` after fitting `model_train`.
This is not the eligibility rule. Eligibility uses `judge_test_id`, and that split is not generated.

| model | fit seconds | state RMSE | one-step RMSE | effluent RMSE | final train loss |
|---|---:|---:|---:|---:|---:|
| hold_last | 0.0 | 1429 | 419.8 | 1.596 | nan |
| linear_narx | 0.2 | 13.65 | 2.526 | 0.2622 | nan |
| lstm_world | 7.5 | 121.9 | 43.8 | 1.283 | 0.0006107 |
| recurrent_world | 15.9 | 176 | 82.79 | 1.685 | 0.004584 |
