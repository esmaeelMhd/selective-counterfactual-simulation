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

Bootstrap of the seed-level mean margin of eligible models, for `calibration_selected_candidate_ranker` only. One seed cannot make an interval. The width limit is 0.10. A row is degenerate when under 1% or over 99% of scenarios are bad. Its margin is fixed by the labels, so it does not count toward the check. `thin_models` counts the eligible models whose own bad share is under or over the same limits. Their margins are fixed by the labels and pull the average toward zero. Degenerate 5% rows: 1. All informative 5% rows narrower than the limit, with one on every system: True.

| system_id | badness_target | n_seeds | bad_share | degenerate | n_models | thin_models | mean_margin | bootstrap_low | bootstrap_high | interval_width | narrower_than_limit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cstr | bad_event | 10 | 0.2268 | False | 3 | 0 | 0.2279 | 0.2007 | 0.2539 | 0.0532 | True |
| cstr | bad_rmse | 10 | 0.4667 | False | 3 | 0 | 0.4621 | 0.4421 | 0.4853 | 0.0431 | True |
| two_tank | bad_event | 10 | 0.0000 | True | 3 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| two_tank | bad_rmse | 10 | 0.6744 | False | 3 | 0 | 0.6850 | 0.6690 | 0.7005 | 0.0315 | True |

## Settings used

These numbers are the working settings in the config. A freeze is a later commit.

- horizon: 32
- ensemble members: 5
- training settings: ensemble_mlp_max_iter 400, gradient_boosting_max_iter 200, mlp_max_iter 400
- `bad_rmse` threshold per system: two_tank 0.08, cstr 0.38, heat_exchanger 0.095
- seeds: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
- minimum judge-test scenarios: 2000

## Descriptive coverages

Descriptive only. These coverages are not part of any decision rule, and nothing was selected or tuned on them. They show how the judges compare where fewer in-distribution scenarios are left to accept. Mean false-accept rate over eligible models and seeds.

| system_id | badness_target | coverage | random | primary | best_simple_on_calibration | support_only | uncertainty_only | disagreement_only | invariant_only | repair_only | event_guard | event_guard_no_invariant | bad_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cstr | bad_event | 0.2500 | 0.2288 | 0.0745 | 0.0760 | 0.0707 | 0.1467 | 0.2221 | 0.0751 | 0.0796 | 0.0941 | 0.1049 | 0.2268 |
| cstr | bad_event | 0.5000 | 0.2280 | 0.1066 | 0.1039 | 0.0992 | 0.1789 | 0.1941 | 0.1024 | 0.1004 | 0.1287 | 0.1642 | 0.2268 |
| cstr | bad_event | 0.7500 | 0.2263 | 0.1832 | 0.1948 | 0.1869 | 0.2183 | 0.2004 | 0.1880 | 0.1916 | 0.1904 | 0.2013 | 0.2268 |
| cstr | bad_rmse | 0.2500 | 0.4628 | 0.0468 | 0.0459 | 0.0627 | 0.3294 | 0.0470 | 0.0468 | 0.0795 | 0.0573 | 0.1310 | 0.4667 |
| cstr | bad_rmse | 0.5000 | 0.4670 | 0.0614 | 0.0784 | 0.2013 | 0.3559 | 0.1060 | 0.0614 | 0.2025 | 0.0735 | 0.1981 | 0.4667 |
| cstr | bad_rmse | 0.7500 | 0.4668 | 0.2889 | 0.2889 | 0.3191 | 0.4080 | 0.2889 | 0.2889 | 0.2889 | 0.2978 | 0.3503 | 0.4667 |
| two_tank | bad_event | 0.2500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| two_tank | bad_event | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| two_tank | bad_event | 0.7500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| two_tank | bad_rmse | 0.2500 | 0.6729 | 0.0733 | 0.1526 | 0.0739 | 0.4635 | 0.3655 | 0.1487 | 0.2000 | 0.1749 | 0.3184 | 0.6744 |
| two_tank | bad_rmse | 0.5000 | 0.6757 | 0.3487 | 0.3990 | 0.3487 | 0.5460 | 0.4683 | 0.3511 | 0.6000 | 0.3760 | 0.4753 | 0.6744 |
| two_tank | bad_rmse | 0.7500 | 0.6741 | 0.5658 | 0.5689 | 0.5658 | 0.6220 | 0.5738 | 0.5658 | 0.7333 | 0.5658 | 0.5878 | 0.6744 |

## Event guard at 5% coverage

Exploratory only. `event_guard` is the v2 candidate: the maximum calibration-percentile rank of invariant, disagreement, and support. `event_guard_no_invariant` replaces invariant with uncertainty, so it needs no full-state signal. `margin_vs_primary` and `margin_vs_calibration_simple` use the simple baseline chosen on calibration rows. They are not a Q3 answer. Only eligible models enter the averages. A guard is not computed when one of its inputs is missing.

| guard | system_id | badness_target | n_models | event_guard_far | primary_far | calibration_simple_far | margin_vs_primary | margin_vs_calibration_simple | room_for_q3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| event_guard | cstr | bad_event | 3 | 0.0102 | 0.0060 | 0.0045 | -0.0043 | -0.0057 | False |
| event_guard | cstr | bad_rmse | 3 | 0.0000 | 0.0000 | 0.0010 | 0.0000 | 0.0010 | False |
| event_guard | two_tank | bad_event | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False |
| event_guard | two_tank | bad_rmse | 3 | 0.0000 | 0.0000 | 0.0003 | 0.0000 | 0.0003 | False |
| event_guard_no_invariant | cstr | bad_event | 3 | 0.0700 | 0.0060 | 0.0045 | -0.0640 | -0.0655 | False |
| event_guard_no_invariant | cstr | bad_rmse | 3 | 0.0510 | 0.0000 | 0.0010 | -0.0510 | -0.0500 | False |
| event_guard_no_invariant | two_tank | bad_event | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False |
| event_guard_no_invariant | two_tank | bad_rmse | 3 | 0.1237 | 0.0000 | 0.0003 | -0.1237 | -0.1233 | False |

`event_guard` has no eligible model, so no row, on: heat_exchanger.

`event_guard_no_invariant` has no eligible model, so no row, on: heat_exchanger.

`room_for_q3` is true only when both the primary ranker and the calibration-chosen simple baseline have a false-accept rate of at least 0.05. The yes rule asks for an absolute improvement of 0.05 over both, and a false-accept rate cannot fall below zero, so where this is false Q3 cannot be yes whatever the guard does.
