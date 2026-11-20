# Exploratory BSM1 model check

Fit on `model_train`, scored on `judge_calibration_id`, one seed (0). No judge-test row exists.
Eligible means the mean per-scenario state RMSE is strictly below both `hold_last` and `linear_narx`, the working rule in `docs/v3/protocol.md`. Stage A pools ten seeds for the ODE systems. This check has one.
Splits read disjoint blocks of the 7 unique dry days, and each window starts from its own minute of one continuous dry run.
Reactor and effluent flow are set from the influent with the open-loop recycle and waste flows, inside each model's own rollout, so recurrent models keep their state.
The calibration split holds 2 independent influent windows, so these numbers are noisy.

| model | fit seconds | mean scenario RMSE | eligible | pooled state RMSE | one-step RMSE | flow | temperature | settler | concentration | effluent RMSE | final train loss |
|---|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hold_last | 0.0 | 96.49 | n/a | 104.1 | 1.047 | 5.422e-12 | 4.152e-15 | 351.6 | 45.72 | 1.885 | nan |
| linear_narx | 0.2 | 21.47 | n/a | 24.69 | 0.3831 | 5.422e-12 | 4.152e-15 | 88.21 | 6.671 | 0.2959 | nan |
| lstm_world | 12.8 | 68.95 | False | 71.45 | 48.32 | 5.422e-12 | 1.343e-08 | 216.3 | 44.59 | 1.412 | 0.007217 |
| recurrent_world | 70.9 | 64.35 | False | 66.51 | 28.66 | 5.422e-12 | 2.649e-08 | 184.4 | 47.91 | 1.063 | 0.01931 |
| mlp_state_space | 10.0 | 45.5 | False | 62.11 | 0.2131 | 5.422e-12 | 4.152e-15 | 223 | 15.51 | 0.8256 | nan |
