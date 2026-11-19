# v3 Stage A report

Label: exploratory. This report does not answer Q1, Q2, or Q3.

The margin is the random false-accept rate minus the judge false-accept rate. A positive margin means the judge accepts fewer bad scenarios than that random ranking. This is not the v2 `absolute_margin`, which selects the lowest test-set false-accept rate among simple and stronger baselines.

## Eligibility

Working rule for this run: mean RMSE on `judge_calibration_id` is strictly below both `hold_last` and `linear_narx` on that system. Eligibility does not read judge-test rows. Learned models passing: 2 system(s) (cstr, two_tank). The plan's Stage A check wants at least two systems.

| system_id | model_id | id_rmse | baseline_hold_last_id_rmse | baseline_linear_narx_id_rmse | eligible |
| --- | --- | --- | --- | --- | --- |
| cstr | ensemble_mlp | 0.0086 | 2.0492 | 0.0483 | True |
| cstr | gradient_boosted_narx | 0.0105 | 2.0492 | 0.0483 | True |
| cstr | hold_last | 2.0492 | 2.0492 | 0.0483 | False |
| cstr | linear_narx | 0.0483 | 2.0492 | 0.0483 | False |
| cstr | mlp_state_space | 0.0110 | 2.0492 | 0.0483 | True |
| heat_exchanger | ensemble_mlp | 0.0222 | 6.7997 | 0.0032 | False |
| heat_exchanger | gradient_boosted_narx | 0.0131 | 6.7997 | 0.0032 | False |
| heat_exchanger | hold_last | 6.7997 | 6.7997 | 0.0032 | False |
| heat_exchanger | linear_narx | 0.0032 | 6.7997 | 0.0032 | False |
| heat_exchanger | mlp_state_space | 0.0256 | 6.7997 | 0.0032 | False |
| two_tank | ensemble_mlp | 0.0022 | 0.4520 | 0.0095 | True |
| two_tank | gradient_boosted_narx | 0.0066 | 0.4520 | 0.0095 | True |
| two_tank | hold_last | 0.4520 | 0.4520 | 0.0095 | False |
| two_tank | linear_narx | 0.0095 | 0.4520 | 0.0095 | False |
| two_tank | mlp_state_space | 0.0024 | 0.4520 | 0.0095 | True |

## Interval at 5% coverage

Bootstrap of the seed-level mean margin of eligible models, for `calibration_selected_candidate_ranker` only. One seed cannot make an interval. The width limit is 0.10. All 5% rows narrower than the limit: True.

| system_id | badness_target | n_seeds | mean_margin | bootstrap_low | bootstrap_high | interval_width | narrower_than_limit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cstr | bad_event | 10 | 0.2264 | 0.1978 | 0.2533 | 0.0555 | True |
| cstr | bad_rmse | 10 | 0.8455 | 0.8383 | 0.8519 | 0.0136 | True |
| two_tank | bad_event | 10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| two_tank | bad_rmse | 10 | 0.7800 | 0.7478 | 0.8035 | 0.0557 | True |

## Settings used

These numbers are Stage A choices. They are not the freeze.

- horizon: 32
- ensemble members: 5
- training iteration cap: mlp 400, ensemble 400, gradient boosting 200
- `bad_rmse` threshold: 0.05
- seeds: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
- minimum judge-test scenarios: 2000

## Event guard at 5% coverage

Exploratory only. The guard is the v2 candidate: the maximum calibration-percentile rank of invariant, disagreement, and support. `margin_vs_primary` and `margin_vs_calibration_simple` use the simple baseline chosen on calibration rows. They are not a Q3 answer. The same guard cannot be used on BSM1, because that system does not provide the invariant signal, and no replacement is written yet.

| system_id | badness_target | event_guard_far | primary_far | calibration_simple_far | margin_vs_primary | margin_vs_calibration_simple |
| --- | --- | --- | --- | --- | --- | --- |
| cstr | bad_event | 0.0550 | 0.0424 | 0.0449 | -0.0126 | -0.0101 |
| cstr | bad_rmse | 0.2023 | 0.2000 | 0.2000 | -0.0023 | -0.0023 |
| heat_exchanger | bad_event | 0.2000 | 0.2000 | 0.2000 | 0.0000 | 0.0000 |
| heat_exchanger | bad_rmse | 0.2013 | 0.2000 | 0.2000 | -0.0013 | -0.0013 |
| two_tank | bad_event | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| two_tank | bad_rmse | 0.1954 | 0.1968 | 0.1950 | 0.0014 | -0.0004 |
