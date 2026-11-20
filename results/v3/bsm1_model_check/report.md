# Exploratory BSM1 model check

Fit on `model_train`, scored on `judge_calibration_id` (the eligibility split) and on `judge_calibration_control`. No judge-test row exists.
Eligible means the mean per-scenario state RMSE, averaged over seeds, is strictly below both `hold_last` and `linear_narx`, the working rule in `docs/v3/protocol.md`. The control column applies the same bar on the control-shift split and does not decide eligibility.
Each influent window has its own flow and load factors. Splits read disjoint blocks of the 7 unique dry days, and each window starts from its own minute of one continuous dry run.
Reactor and effluent flow are set from the influent with the open-loop recycle and waste flows, inside each model's own rollout, so recurrent models keep their state.
The LSTM and recurrent adapters predict the change in state over one step. The calibration split holds 2 independent daily influent shapes, so these numbers are noisy.
The LSTM epoch count (40) was chosen after seeing this check on the eligibility split with 15 and 40 epochs. The eligibility result is optimistic by that choice. At 80 epochs one of two seeds fails it.

| model | seeds | fit seconds | mean scenario RMSE | seeds below bar | eligible | control-shift RMSE | control below bar | pooled state RMSE | one-step RMSE | settler | concentration | effluent RMSE | final train loss |
|---|---:|---:|---:|---:|:---:|---:|:---:|---:|---:|---:|---:|---:|---:|
| hold_last | 1 | 0.0 | 96.49 | n/a | n/a | 83.47 | n/a | 104.1 | 1.047 | 351.6 | 45.72 | 1.885 | nan |
| linear_narx | 1 | 0.2 | 21.47 | n/a | n/a | 78.84 | n/a | 24.69 | 0.3831 | 88.21 | 6.671 | 0.2959 | nan |
| mlp_state_space | 3 | 10.5 | 57.24 | 0/3 | False | 118.6 | False | 68.49 | 0.2261 | 243.9 | 19.3 | 1.205 | nan |
| lstm_world | 3 | 30.6 | 9.547 | 3/3 | True | 36.47 | True | 10.89 | 0.6438 | 38.64 | 3.187 | 0.3482 | 0.01337 |
| ensemble_lstm | 3 | 228.0 | 7.618 | 3/3 | True | 24.01 | True | 8.381 | 0.612 | 29.82 | 2.365 | 0.3031 | nan |
| recurrent_world | 1 | 203.3 | 27.95 | 0/1 | False | 61.92 | True | 30.15 | 0.8041 | 107 | 8.899 | 0.718 | 0.04035 |
