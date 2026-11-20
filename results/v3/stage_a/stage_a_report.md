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

Bootstrap of the seed-level mean margin of eligible models, for `calibration_selected_candidate_ranker` only. One seed cannot make an interval. The width limit is 0.10. A row is degenerate when under 1% or over 99% of scenarios are bad. Its margin is fixed by the labels, so it does not count toward the check. Degenerate 5% rows: 1. All informative 5% rows narrower than the limit, with one on every system: True.

| system_id | badness_target | n_seeds | bad_share | degenerate | mean_margin | bootstrap_low | bootstrap_high | interval_width | narrower_than_limit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cstr | bad_event | 10 | 0.2268 | False | 0.2264 | 0.1978 | 0.2533 | 0.0555 | True |
| cstr | bad_rmse | 10 | 0.4667 | False | 0.4621 | 0.4421 | 0.4853 | 0.0431 | True |
| two_tank | bad_event | 10 | 0.0000 | True | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| two_tank | bad_rmse | 10 | 0.6744 | False | 0.6850 | 0.6690 | 0.7005 | 0.0315 | True |

## Settings used

These numbers are Stage A choices. They are not the freeze.

- horizon: 32
- ensemble members: 5
- training settings: ensemble_mlp_max_iter 400, gradient_boosting_max_iter 200, mlp_max_iter 400
- `bad_rmse` threshold per system: two_tank 0.08, cstr 0.38, heat_exchanger 0.095
- seeds: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
- minimum judge-test scenarios: 2000

## Event guard at 5% coverage

Exploratory only. The guard is the v2 candidate: the maximum calibration-percentile rank of invariant, disagreement, and support. `margin_vs_primary` and `margin_vs_calibration_simple` use the simple baseline chosen on calibration rows. They are not a Q3 answer. Only eligible models enter the averages. The guard is not computed when one of its inputs is missing. On BSM1 that is the invariant signal, and no replacement guard is written yet.

| system_id | badness_target | n_models | event_guard_far | primary_far | calibration_simple_far | margin_vs_primary | margin_vs_calibration_simple |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cstr | bad_event | 3 | 0.0102 | 0.0074 | 0.0045 | -0.0029 | -0.0057 |
| cstr | bad_rmse | 3 | 0.0000 | 0.0000 | 0.0010 | 0.0000 | 0.0010 |
| two_tank | bad_event | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| two_tank | bad_rmse | 3 | 0.0000 | 0.0000 | 0.0003 | 0.0000 | 0.0003 |

No eligible model, so no guard row: heat_exchanger.
