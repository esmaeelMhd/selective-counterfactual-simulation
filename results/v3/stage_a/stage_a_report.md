# v3 Stage A report

Label: exploratory. This report does not answer Q1, Q2, or Q3.

The margin is the random false-accept rate minus the judge false-accept rate. A positive margin means the judge accepts fewer bad scenarios than that random ranking. This is not the v2 `absolute_margin`, which selects the lowest test-set false-accept rate among simple and stronger baselines.

## Provenance

- git_commit: `2b325166aff133376020532c061aa6907fe239f5`
- git_dirty: `False`
- protocol_sha256: `b26ea01e69954cdfcecb747c1057a1ffb0462eb630e9916fd6a6bc86a0a47253`
- rmse_metric: two_tank raw, cstr raw, heat_exchanger raw

## Eligibility

Working rule for this run: mean RMSE on `judge_calibration_id` is strictly below `hold_last` and `linear_narx` on that system. Eligibility does not read judge-test rows. Learned models passing: 2 system(s) (cstr, two_tank). The plan's Stage A check wants at least two systems.

| system_id | model_id | id_rmse | baseline_hold_last_id_rmse | baseline_linear_narx_id_rmse | eligible |
| --- | --- | --- | --- | --- | --- |
| two_tank | ensemble_mlp | 0.0022 | 0.4520 | 0.0095 | True |
| two_tank | gradient_boosted_narx | 0.0066 | 0.4520 | 0.0095 | True |
| two_tank | hold_last | 0.4520 | 0.4520 | 0.0095 | False |
| two_tank | linear_narx | 0.0095 | 0.4520 | 0.0095 | False |
| two_tank | mlp_state_space | 0.0024 | 0.4520 | 0.0095 | True |
| cstr | ensemble_mlp | 0.0083 | 2.0492 | 0.0483 | True |
| cstr | gradient_boosted_narx | 0.0105 | 2.0492 | 0.0483 | True |
| cstr | hold_last | 2.0492 | 2.0492 | 0.0483 | False |
| cstr | linear_narx | 0.0483 | 2.0492 | 0.0483 | False |
| cstr | mlp_state_space | 0.0110 | 2.0492 | 0.0483 | True |
| heat_exchanger | ensemble_mlp | 0.0226 | 6.7997 | 0.0032 | False |
| heat_exchanger | gradient_boosted_narx | 0.0131 | 6.7997 | 0.0032 | False |
| heat_exchanger | hold_last | 6.7997 | 6.7997 | 0.0032 | False |
| heat_exchanger | linear_narx | 0.0032 | 6.7997 | 0.0032 | False |
| heat_exchanger | mlp_state_space | 0.0256 | 6.7997 | 0.0032 | False |

## Primary coverages (Q1 and Q2)

Bootstrap of the seed-level mean margin of eligible models, for `calibration_selected_candidate_ranker` only, separately at every primary coverage. `mean_relative_margin` is the ratio of means (`mean_margin / mean_random_far`), matching Q3. One seed cannot make an interval. The width limit at 5% is 0.10. A row is degenerate when under 1% or over 99% of scenarios are bad. Its margin is fixed by the labels, so it does not count toward the check. `thin_models` counts the eligible models whose own bad share is under or over the same limits. Their margins are fixed by the labels and pull the average toward zero. Degenerate 5% rows: 1. All informative 5% rows narrower than the limit, with one on every system: True.

| system_id | badness_target | coverage | n_seeds | bad_share | degenerate | n_models | thin_models | mean_margin | mean_relative_margin | bootstrap_low | bootstrap_high | interval_width | narrower_than_limit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cstr | bad_event | 0.0500 | 10 | 0.2215 | False | 3 | 0 | 0.2214 | 0.9718 | 0.1971 | 0.2461 | 0.0489 | True |
| cstr | bad_event | 0.1000 | 10 | 0.2215 | False | 3 | 0 | 0.2183 | 0.9627 | 0.1981 | 0.2419 | 0.0438 | True |
| cstr | bad_rmse | 0.0500 | 10 | 0.4654 | False | 3 | 0 | 0.4612 | 1.0000 | 0.4420 | 0.4827 | 0.0407 | True |
| cstr | bad_rmse | 0.1000 | 10 | 0.4654 | False | 3 | 0 | 0.4654 | 1.0000 | 0.4523 | 0.4785 | 0.0261 | True |
| two_tank | bad_event | 0.0500 | 10 | 0.0000 | True | 3 | 3 | 0.0000 |  | 0.0000 | 0.0000 | 0.0000 | True |
| two_tank | bad_event | 0.1000 | 10 | 0.0000 | True | 3 | 3 | 0.0000 |  | 0.0000 | 0.0000 | 0.0000 | True |
| two_tank | bad_rmse | 0.0500 | 10 | 0.6747 | False | 3 | 0 | 0.6853 | 1.0000 | 0.6692 | 0.7008 | 0.0317 | True |
| two_tank | bad_rmse | 0.1000 | 10 | 0.6747 | False | 3 | 0 | 0.6792 | 1.0000 | 0.6677 | 0.6917 | 0.0239 | True |

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
| cstr | bad_event | 0.2500 | 0.2242 | 0.0477 | 0.0561 | 0.0706 | 0.0606 | 0.0395 | 0.0750 | 0.2215 | 0.0405 | 0.0260 | 0.2215 |
| cstr | bad_event | 0.5000 | 0.2229 | 0.0938 | 0.0904 | 0.0988 | 0.1032 | 0.1175 | 0.0977 | 0.2215 | 0.1190 | 0.1298 | 0.2215 |
| cstr | bad_event | 0.7500 | 0.2212 | 0.1754 | 0.1798 | 0.1825 | 0.2171 | 0.1907 | 0.1807 | 0.2215 | 0.1882 | 0.2138 | 0.2215 |
| cstr | bad_rmse | 0.2500 | 0.4616 | 0.0060 | 0.0154 | 0.0606 | 0.0077 | 0.0140 | 0.0459 | 0.4654 | 0.0324 | 0.0016 | 0.4654 |
| cstr | bad_rmse | 0.5000 | 0.4660 | 0.0450 | 0.0489 | 0.1995 | 0.0767 | 0.0514 | 0.0605 | 0.4654 | 0.0710 | 0.0166 | 0.4654 |
| cstr | bad_rmse | 0.7500 | 0.4657 | 0.2872 | 0.2872 | 0.3174 | 0.2467 | 0.2872 | 0.2872 | 0.4654 | 0.2873 | 0.2467 | 0.4654 |
| two_tank | bad_event | 0.2500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| two_tank | bad_event | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| two_tank | bad_event | 0.7500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| two_tank | bad_rmse | 0.2500 | 0.6734 | 0.0735 | 0.0737 | 0.0740 | 0.0414 | 0.1407 | 0.1477 | 0.6747 | 0.0890 | 0.0302 | 0.6747 |
| two_tank | bad_rmse | 0.5000 | 0.6761 | 0.3495 | 0.3495 | 0.3495 | 0.2451 | 0.4217 | 0.3500 | 0.6747 | 0.3520 | 0.2642 | 0.6747 |
| two_tank | bad_rmse | 0.7500 | 0.6745 | 0.5663 | 0.5663 | 0.5663 | 0.4683 | 0.5663 | 0.5663 | 0.6747 | 0.5663 | 0.4683 | 0.6747 |

## Event guard at primary coverages (Q3)

Exploratory only. `event_guard` is the v2 candidate: the maximum calibration-percentile rank of invariant, disagreement, and support. `event_guard_no_invariant` replaces invariant with uncertainty, so it needs no full-state signal. `margin_vs_primary` and `margin_vs_calibration_simple` use the simple baseline chosen on calibration rows. Only eligible models enter the averages. A guard is not computed when one of its inputs is missing.

| guard | system_id | badness_target | coverage | n_models | event_guard_far | primary_far | calibration_simple_far | margin_vs_primary | rel_vs_primary | ci_low_vs_primary | ci_high_vs_primary | margin_vs_calibration_simple | rel_vs_simple | ci_low_vs_simple | ci_high_vs_simple | room_for_q3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| event_guard | cstr | bad_event | 0.0500 | 3 | 0.0081 | 0.0064 | 0.0057 | -0.0017 | -0.2623 | -0.0051 | 0.0012 | -0.0024 | -0.4201 | -0.0051 | -0.0001 | False |
| event_guard | cstr | bad_event | 0.1000 | 3 | 0.0067 | 0.0085 | 0.0083 | 0.0018 | 0.2111 | 0.0004 | 0.0036 | 0.0017 | 0.1999 | -0.0000 | 0.0035 | False |
| event_guard | cstr | bad_rmse | 0.0500 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  | 0.0000 | 0.0000 | 0.0000 |  | 0.0000 | 0.0000 | False |
| event_guard | cstr | bad_rmse | 0.1000 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  | 0.0000 | 0.0000 | 0.0000 |  | 0.0000 | 0.0000 | False |
| event_guard | two_tank | bad_event | 0.0500 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  | 0.0000 | 0.0000 | 0.0000 |  | 0.0000 | 0.0000 | False |
| event_guard | two_tank | bad_event | 0.1000 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  | 0.0000 | 0.0000 | 0.0000 |  | 0.0000 | 0.0000 | False |
| event_guard | two_tank | bad_rmse | 0.0500 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  | 0.0000 | 0.0000 | 0.0000 |  | 0.0000 | 0.0000 | False |
| event_guard | two_tank | bad_rmse | 0.1000 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  | 0.0000 | 0.0000 | 0.0000 |  | 0.0000 | 0.0000 | False |
| event_guard_no_invariant | cstr | bad_event | 0.0500 | 1 | 0.0050 | 0.0036 | 0.0029 | -0.0014 | -0.4000 | -0.0050 | 0.0021 | -0.0021 | -0.7500 | -0.0057 | 0.0007 | False |
| event_guard_no_invariant | cstr | bad_event | 0.1000 | 1 | 0.0050 | 0.0057 | 0.0057 | 0.0007 | 0.1250 | -0.0007 | 0.0021 | 0.0007 | 0.1250 | -0.0007 | 0.0021 | False |
| event_guard_no_invariant | cstr | bad_rmse | 0.0500 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  | 0.0000 | 0.0000 | 0.0000 |  | 0.0000 | 0.0000 | False |
| event_guard_no_invariant | cstr | bad_rmse | 0.1000 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  | 0.0000 | 0.0000 | 0.0000 |  | 0.0000 | 0.0000 | False |
| event_guard_no_invariant | two_tank | bad_event | 0.0500 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  | 0.0000 | 0.0000 | 0.0000 |  | 0.0000 | 0.0000 | False |
| event_guard_no_invariant | two_tank | bad_event | 0.1000 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  | 0.0000 | 0.0000 | 0.0000 |  | 0.0000 | 0.0000 | False |
| event_guard_no_invariant | two_tank | bad_rmse | 0.0500 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  | 0.0000 | 0.0000 | 0.0000 |  | 0.0000 | 0.0000 | False |
| event_guard_no_invariant | two_tank | bad_rmse | 0.1000 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  | 0.0000 | 0.0000 | 0.0000 |  | 0.0000 | 0.0000 | False |

`event_guard` has no eligible model, so no row, on: heat_exchanger.

`event_guard_no_invariant` has no eligible model, so no row, on: heat_exchanger.

`room_for_q3` is true only when both the primary ranker and the calibration-chosen simple baseline have a false-accept rate of at least 0.05. The yes rule asks for an absolute improvement of 0.05 over both, and a false-accept rate cannot fall below zero, so where this is false Q3 cannot be yes whatever the guard does.
