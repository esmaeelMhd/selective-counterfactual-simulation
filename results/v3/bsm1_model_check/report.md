# Exploratory BSM1 model check

Scored on `judge_calibration_id` after fitting `model_train`.
`judge_calibration_id` is the eligibility split, but this is not the eligibility rule: RMSE is pooled over rows rather than averaged per scenario.
Splits read disjoint blocks of the dry file, and each window starts from its own minute of one continuous dry run.
Reactor and effluent flow are set from the influent with the open-loop recycle and waste flows.

| model | fit seconds | state RMSE | one-step RMSE | flow | temperature | settler | concentration | effluent RMSE | final train loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hold_last | 0.0 | 108.2 | 1.137 | 4.618e-13 | 0 | 363.6 | 48.73 | 2.413 | nan |
| linear_narx | 0.9 | 21.65 | 0.4662 | 4.618e-13 | 0 | 77.66 | 5.501 | 0.5444 | nan |
| lstm_world | 15.6 | 100.9 | 7.056 | 4.618e-13 | 8.947e-10 | 342 | 43.39 | 2 | 0.000519 |
| recurrent_world | 39.5 | 81.88 | 6.863 | 4.618e-13 | 0 | 281.6 | 32.25 | 1.9 | 0.003648 |
| mlp_state_space | 14.4 | 94.46 | 0.3318 | 4.618e-13 | 0.1427 | 340.2 | 22.16 | 0.8281 | nan |
