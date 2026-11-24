# Exploratory BSM1 model check

Fit on `model_train`, scored on `judge_calibration_id` (the eligibility split) and on `judge_calibration_control`. No judge-test row exists.
Eligible means the mean standardized per-scenario state RMSE (z-scored over non-constant channels using `model_train` std only), averaged over seeds, is strictly below both `hold_last` and `linear_narx`. Raw state RMSE is also reported for comparison. The control column applies the same bar on the control-shift split and does not decide eligibility.
Each influent window has its own flow and load factors. Splits read disjoint blocks of the 7 unique dry days, and each window starts from its own minute of one continuous dry run. The 15-minute history window evolves at nominal aeration; counterfactual aeration begins at t=0.
Reactor and effluent flow are set from the influent with the open-loop recycle and waste flows, inside each model's own rollout, so recurrent models keep their state.
The LSTM and recurrent adapters predict the change in state over one step. The calibration split holds 2 independent daily influent shapes, so these numbers are noisy.
The LSTM, its ensemble and the recurrent model train under a stopping rule fixed before any calibration row was read: the last 15% of the `model_train` windows are held out, less a gap of 18 windows, the free-running rollout error on them is checked every 5 epochs, the best weights are kept, and training stops after 3 checks without improvement or at 120 epochs. Neither calibration split decides a budget. The delta MLP keeps its earlier 80 iterations.

Calibration-derived standardized `bad_rmse` threshold: **0.5069** (3x the 99th percentile 0.1690 of `ensemble_lstm` on `judge_calibration_id`; max eligible ID scenario 0.3957; marks 44.1% of eligible calibration scenarios as bad).

| model | seeds | fit seconds | std scenario RMSE | seeds below bar | eligible | std control RMSE | control below bar | raw scenario RMSE | one-step RMSE | settler | concentration | effluent RMSE | epochs used |
|---|---:|---:|---:|---:|:---:|---:|:---:|---:|---:|---:|---:|---:|---:|
| hold_last | 1 | 0.0 | 0.8636 | n/a | n/a | 1.151 | n/a | 96.49 | 1.048 | 351.6 | 45.72 | 1.885 | nan |
| linear_narx | 1 | 0.2 | 0.2426 | n/a | n/a | 0.5544 | n/a | 23.14 | 0.389 | 96.61 | 7.587 | 0.2903 | nan |
| mlp_state_space | 3 | 11.4 | 1.344 | 0/3 | False | 2.54 | False | 96.96 | 0.3278 | 433 | 27.18 | 1.252 | nan |
| lstm_world | 3 | 44.0 | 0.1538 | 3/3 | True | 0.8338 | False | 15.16 | 0.696 | 57.82 | 5.513 | 0.5883 | 50 |
| ensemble_lstm | 3 | 228.3 | 0.1015 | 3/3 | True | 0.6401 | False | 9.998 | 0.6664 | 38.51 | 3.621 | 0.5264 | 52.67 |
| recurrent_world | 1 | 169.5 | 0.3143 | 0/1 | False | 0.8437 | False | 28.52 | 0.7879 | 105 | 10.96 | 0.8315 | 20 |
