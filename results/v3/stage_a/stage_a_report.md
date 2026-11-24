# v3 Stage A report

Label: exploratory. This report does not answer Q1, Q2, or Q3.

The margin is the random false-accept rate minus the judge false-accept rate. A positive margin means the judge accepts fewer bad scenarios than that random ranking. This is not the v2 `absolute_margin`, which selects the lowest test-set false-accept rate among simple and stronger baselines.

## Provenance

- git_commit: `b880b70760a55d3070a05377d9143b7246f202e7`
- protocol_sha256: `0d15e01b78fe4c1d4e6019af8eb0dd23bbbd1af2bccd1ed042c1489cfc5bdffb`
- rmse_metric: two_tank raw, cstr raw, heat_exchanger raw

## Eligibility

Working rule for this run: mean RMSE on `judge_calibration_id` is strictly below both `hold_last` and `linear_narx` on that system. Eligibility does not read judge-test rows. Learned models passing: 2 system(s) (cstr, two_tank). The plan's Stage A check wants at least two systems.

| system_id | model_id | id_rmse | baseline_hold_last_id_rmse | baseline_linear_narx_id_rmse | eligible |
| --- | --- | --- | --- | --- | --- |
| two_tank | ensemble_mlp | 0.0022 | 0.4520 | 0.0095 | True |
| two_tank | gradient_boosted_narx | 0.0066 | 0.4520 | 0.0095 | True |
| two_tank | hold_last | 0.4520 | 0.4520 | 0.0095 | False |
| two_tank | linear_narx | 0.0095 | 0.4520 | 0.0095 | False |
| two_tank | mlp_state_space | 0.0024 | 0.4520 | 0.0095 | True |
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

## Primary coverages (Q1 and Q2)

Bootstrap of the seed-level mean margin of eligible models, for `calibration_selected_candidate_ranker` only, separately at every primary coverage. One seed cannot make an interval. The width limit at 5% is 0.10. A row is degenerate when under 1% or over 99% of scenarios are bad. Its margin is fixed by the labels, so it does not count toward the check. `thin_models` counts the eligible models whose own bad share is under or over the same limits. Their margins are fixed by the labels and pull the average toward zero. Degenerate 5% rows: 1. All informative 5% rows narrower than the limit, with one on every system: True.

| system_id | badness_target | coverage | n_seeds | bad_share | degenerate | n_models | thin_models | mean_margin | mean_relative_margin | bootstrap_low | bootstrap_high | interval_width | narrower_than_limit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cstr | bad_event | 0.0500 | 10 | 0.2268 | False | 3 | 0 | 0.2260 | 0.9552 | 0.1973 | 0.2537 | 0.0564 | True |
| cstr | bad_event | 0.1000 | 10 | 0.2268 | False | 3 | 0 | 0.2223 | 0.9507 | 0.1988 | 0.2479 | 0.0491 | True |
| cstr | bad_rmse | 0.0500 | 10 | 0.4667 | False | 3 | 0 | 0.4621 | 1.0000 | 0.4421 | 0.4853 | 0.0431 | True |
| cstr | bad_rmse | 0.1000 | 10 | 0.4667 | False | 3 | 0 | 0.4663 | 1.0000 | 0.4524 | 0.4807 | 0.0283 | True |
| two_tank | bad_event | 0.0500 | 10 | 0.0000 | True | 3 | 3 | 0.0000 |  | 0.0000 | 0.0000 | 0.0000 | True |
| two_tank | bad_event | 0.1000 | 10 | 0.0000 | True | 3 | 3 | 0.0000 |  | 0.0000 | 0.0000 | 0.0000 | True |
| two_tank | bad_rmse | 0.0500 | 10 | 0.6744 | False | 3 | 0 | 0.6850 | 1.0000 | 0.6690 | 0.7005 | 0.0315 | True |
| two_tank | bad_rmse | 0.1000 | 10 | 0.6744 | False | 3 | 0 | 0.6790 | 1.0000 | 0.6677 | 0.6916 | 0.0239 | True |

## Settings used

These numbers are the working settings in the config. A freeze is a later commit.

- horizon: 32
- ensemble members: 5
- training settings: ensemble_mlp_max_iter 400, gradient_boosting_max_iter 200, mlp_max_iter 400
- `bad_rmse` threshold per system: two_tank 0.08, cstr 0.38, heat_exchanger 0.095
- `rmse_metric` per system: two_tank raw, cstr raw, heat_exchanger raw
- seeds: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
- minimum judge-test scenarios: 2000

## Descriptive coverages

Descriptive only. These coverages are not part of any decision rule, and nothing was selected or tuned on them. They show how the judges compare where fewer in-distribution scenarios are left to accept. Each column averages over eligible models for which that judge is defined (for example, `uncertainty_only` and `event_guard_no_invariant` cover models with native epistemic uncertainty only).

| system_id | badness_target | coverage | random | primary | best_simple_on_calibration | support_only | uncertainty_only | disagreement_only | invariant_only | repair_only | event_guard | event_guard_no_invariant | bad_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cstr | bad_event | 0.2500 | 0.2288 | 0.0362 | 0.0475 | 0.0707 | 0.0250 | 0.0473 | 0.0751 | 0.0796 | 0.0438 | 0.0307 | 0.2268 |
| cstr | bad_event | 0.5000 | 0.2280 | 0.0904 | 0.0900 | 0.0992 | 0.1171 | 0.1297 | 0.1024 | 0.1004 | 0.1229 | 0.1397 | 0.2268 |
| cstr | bad_event | 0.7500 | 0.2263 | 0.1817 | 0.1871 | 0.1869 | 0.2372 | 0.1973 | 0.1880 | 0.1916 | 0.1957 | 0.2346 | 0.2268 |
| cstr | bad_rmse | 0.2500 | 0.4628 | 0.0072 | 0.0108 | 0.0627 | 0.0034 | 0.0113 | 0.0468 | 0.0795 | 0.0356 | 0.0001 | 0.4667 |
| cstr | bad_rmse | 0.5000 | 0.4670 | 0.0478 | 0.0525 | 0.2013 | 0.0896 | 0.0533 | 0.0614 | 0.2025 | 0.0727 | 0.0236 | 0.4667 |
| cstr | bad_rmse | 0.7500 | 0.4668 | 0.2889 | 0.2889 | 0.3191 | 0.2518 | 0.2889 | 0.2889 | 0.2889 | 0.2890 | 0.2519 | 0.4667 |
| two_tank | bad_event | 0.2500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| two_tank | bad_event | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| two_tank | bad_event | 0.7500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| two_tank | bad_rmse | 0.2500 | 0.6729 | 0.0735 | 0.0737 | 0.0739 | 0.0730 | 0.1475 | 0.1487 | 0.2000 | 0.0878 | 0.0436 | 0.6744 |
| two_tank | bad_rmse | 0.5000 | 0.6757 | 0.3487 | 0.3487 | 0.3487 | 0.2752 | 0.4310 | 0.3511 | 0.6000 | 0.3518 | 0.2786 | 0.6744 |
| two_tank | bad_rmse | 0.7500 | 0.6741 | 0.5658 | 0.5658 | 0.5658 | 0.4667 | 0.5658 | 0.5658 | 0.7333 | 0.5658 | 0.4667 | 0.6744 |

## Event guard at primary coverages (Q3)

Exploratory only. `event_guard` is the v2 candidate: the maximum calibration-percentile rank of invariant, disagreement, and support. `event_guard_no_invariant` replaces invariant with uncertainty, so it needs no full-state signal. `margin_vs_primary` and `margin_vs_calibration_simple` use the simple baseline chosen on calibration rows. Only eligible models enter the averages. A guard is not computed when one of its inputs is missing.

| guard | system_id | badness_target | coverage | n_models | event_guard_far | primary_far | calibration_simple_far | margin_vs_primary | rel_vs_primary | ci_low_vs_primary | ci_high_vs_primary | margin_vs_calibration_simple | rel_vs_simple | ci_low_vs_simple | ci_high_vs_simple | room_for_q3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| event_guard | cstr | bad_event | 0.0500 | 3 | 0.0079 | 0.0079 | 0.0071 | -0.0000 | -0.0000 | -0.0027 | 0.0026 | -0.0007 | -0.1000 | -0.0033 | 0.0014 | False |
| event_guard | cstr | bad_event | 0.1000 | 3 | 0.0065 | 0.0087 | 0.0086 | 0.0021 | 0.2466 | 0.0008 | 0.0038 | 0.0020 | 0.2361 | 0.0007 | 0.0037 | False |
| event_guard | cstr | bad_rmse | 0.0500 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  | 0.0000 | 0.0000 | 0.0000 |  | 0.0000 | 0.0000 | False |
| event_guard | cstr | bad_rmse | 0.1000 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  | 0.0000 | 0.0000 | 0.0000 |  | 0.0000 | 0.0000 | False |
| event_guard | two_tank | bad_event | 0.0500 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  | 0.0000 | 0.0000 | 0.0000 |  | 0.0000 | 0.0000 | False |
| event_guard | two_tank | bad_event | 0.1000 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  | 0.0000 | 0.0000 | 0.0000 |  | 0.0000 | 0.0000 | False |
| event_guard | two_tank | bad_rmse | 0.0500 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  | 0.0000 | 0.0000 | 0.0000 |  | 0.0000 | 0.0000 | False |
| event_guard | two_tank | bad_rmse | 0.1000 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  | 0.0000 | 0.0000 | 0.0000 |  | 0.0000 | 0.0000 | False |
| event_guard_no_invariant | cstr | bad_event | 0.0500 | 1 | 0.0043 | 0.0050 | 0.0043 | 0.0007 | 0.1429 | -0.0036 | 0.0050 | 0.0000 | 0.0000 | -0.0036 | 0.0036 | False |
| event_guard_no_invariant | cstr | bad_event | 0.1000 | 1 | 0.0050 | 0.0057 | 0.0057 | 0.0007 | 0.1250 | 0.0000 | 0.0018 | 0.0007 | 0.1250 | 0.0000 | 0.0018 | False |
| event_guard_no_invariant | cstr | bad_rmse | 0.0500 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  | 0.0000 | 0.0000 | 0.0000 |  | 0.0000 | 0.0000 | False |
| event_guard_no_invariant | cstr | bad_rmse | 0.1000 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  | 0.0000 | 0.0000 | 0.0000 |  | 0.0000 | 0.0000 | False |
| event_guard_no_invariant | two_tank | bad_event | 0.0500 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  | 0.0000 | 0.0000 | 0.0000 |  | 0.0000 | 0.0000 | False |
| event_guard_no_invariant | two_tank | bad_event | 0.1000 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  | 0.0000 | 0.0000 | 0.0000 |  | 0.0000 | 0.0000 | False |
| event_guard_no_invariant | two_tank | bad_rmse | 0.0500 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  | 0.0000 | 0.0000 | 0.0000 |  | 0.0000 | 0.0000 | False |
| event_guard_no_invariant | two_tank | bad_rmse | 0.1000 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  | 0.0000 | 0.0000 | 0.0000 |  | 0.0000 | 0.0000 | False |

`event_guard` has no eligible model, so no row, on: heat_exchanger.

`event_guard_no_invariant` has no eligible model, so no row, on: heat_exchanger.

`room_for_q3` is true only when both the primary ranker and the calibration-chosen simple baseline have a false-accept rate of at least 0.05. The yes rule asks for an absolute improvement of 0.05 over both, and a false-accept rate cannot fall below zero, so where this is false Q3 cannot be yes whatever the guard does.
