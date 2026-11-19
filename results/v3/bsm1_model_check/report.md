# Exploratory BSM1 model check

Fit on `model_train`, scored on `judge_calibration_id`, one seed (0). No judge-test row exists.
Eligible means the mean per-scenario state RMSE is strictly below both `hold_last` and `linear_narx`, the working rule in `docs/v3/protocol.md`. Stage A pools ten seeds for the ODE systems. This check has one.
Splits read disjoint blocks of the 7 unique dry days, and each window starts from its own minute of one continuous dry run.
Reactor and effluent flow are set from the influent with the open-loop recycle and waste flows, inside each model's own rollout, so recurrent models keep their state.
The calibration split holds 2 independent influent windows, so these numbers are noisy.

| model | fit seconds | mean scenario RMSE | eligible | pooled state RMSE | one-step RMSE | flow | temperature | settler | concentration | effluent RMSE | final train loss |
|---|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hold_last | 0.0 | 95.69 | n/a | 103.2 | 0.947 | 3.836e-13 | 0 | 348.3 | 45.39 | 1.843 | nan |
| linear_narx | 1.5 | 30.4 | n/a | 35.56 | 0.3272 | 3.836e-13 | 0 | 127.3 | 9.389 | 0.2938 | nan |
| lstm_world | 42.5 | 27.42 | True | 28.36 | 48.23 | 3.836e-13 | 9.672e-09 | 81.1 | 19.55 | 0.5818 | 0.00387 |
| recurrent_world | 154.6 | 42.14 | False | 46.83 | 29.81 | 3.836e-13 | 2.073e-08 | 159.4 | 19.66 | 1.068 | 0.01126 |
| mlp_state_space | 9.1 | 21.08 | True | 22.08 | 0.2407 | 3.836e-13 | 1.821e-07 | 79.39 | 5.38 | 0.3847 | nan |
