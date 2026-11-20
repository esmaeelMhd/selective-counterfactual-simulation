# Exploratory BSM1 model check

Fit on `model_train`, scored on `judge_calibration_id` (the eligibility split) and on `judge_calibration_control`. No judge-test row exists.
Eligible means the mean per-scenario state RMSE, averaged over seeds, is strictly below both `hold_last` and `linear_narx`, the working rule in `docs/v3/protocol.md`. The control column applies the same bar on the control-shift split and does not decide eligibility.
Each influent window has its own flow and load factors. Splits read disjoint blocks of the 7 unique dry days, and each window starts from its own minute of one continuous dry run.
Reactor and effluent flow are set from the influent with the open-loop recycle and waste flows, inside each model's own rollout, so recurrent models keep their state.
The LSTM and recurrent adapters predict the change in state over one step. The calibration split holds 2 independent daily influent shapes, so these numbers are noisy.
The LSTM, its ensemble and the recurrent model train under a stopping rule fixed before any calibration row was read: the last 15% of the `model_train` windows are held out, less a gap of 18 windows, the free-running rollout error on them is checked every 5 epochs, the best weights are kept, and training stops after 3 checks without improvement or at 120 epochs. Neither calibration split decides a budget. The delta MLP keeps its earlier 80 iterations.

| model | seeds | fit seconds | mean scenario RMSE | seeds below bar | eligible | control-shift RMSE | control below bar | pooled state RMSE | one-step RMSE | settler | concentration | effluent RMSE | epochs used |
|---|---:|---:|---:|---:|:---:|---:|:---:|---:|---:|---:|---:|---:|---:|
| hold_last | 1 | 0.0 | 96.49 | n/a | n/a | 83.47 | n/a | 104.1 | 1.047 | 351.6 | 45.72 | 1.885 | nan |
| linear_narx | 1 | 0.2 | 21.47 | n/a | n/a | 78.84 | n/a | 24.69 | 0.3831 | 88.21 | 6.671 | 0.2959 | nan |
| mlp_state_space | 3 | 9.9 | 57.24 | 0/3 | False | 118.6 | False | 68.49 | 0.2261 | 243.9 | 19.3 | 1.205 | nan |
| lstm_world | 3 | 39.7 | 15.68 | 3/3 | True | 51.37 | True | 16.94 | 0.6527 | 60.15 | 4.929 | 0.6228 | 55 |
| ensemble_lstm | 3 | 202.7 | 9.307 | 3/3 | True | 31.07 | True | 10.29 | 0.6435 | 36.1 | 3.455 | 0.4788 | 52.67 |
| recurrent_world | 1 | 151.1 | 26.31 | 0/1 | False | 54.87 | True | 27.35 | 0.7433 | 94.85 | 10.15 | 0.8476 | 25 |
