# Exploratory BSM1 model check

Scored on `judge_calibration_id` after fitting `model_train`.
This is not the eligibility rule. Eligibility uses `judge_test_id`, and that split is not generated.

| model | fit seconds | state RMSE | effluent RMSE | final train loss |
|---|---:|---:|---:|---:|
| hold_last | 0.0 | 1429 | 1.596 | nan |
| linear_narx | 0.2 | 13.65 | 0.2622 | nan |
| lstm_world | 9.0 | 961 | 1.378 | 0.4434 |
| recurrent_world | 71.9 | 864.6 | 1.499 | 0.5558 |
