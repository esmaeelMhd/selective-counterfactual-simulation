# Smoke Report

## Experiment config

```json
{
  "bad_threshold": {
    "metric": "rmse",
    "value": 0.8
  },
  "coverages": [
    0.1,
    0.2,
    0.4,
    0.6,
    0.8,
    1.0
  ],
  "dt": 0.1,
  "experiment_id": "smoke_cstr",
  "horizon": 50,
  "judges": [
    "support_only",
    "uncertainty_only",
    "disagreement_only",
    "invariant_only",
    "repair_only",
    "combined_linear",
    "random_baseline",
    "oracle_error_rank"
  ],
  "models": [
    "hold_last",
    "linear_narx",
    "mlp_state_space"
  ],
  "n_id_test": 40,
  "n_ood_test": 40,
  "n_train": 180,
  "output_dir": "results/smoke_cstr",
  "seed": 42,
  "system_id": "cstr",
  "uncertainty_samples": 6
}
```

## Dataset summary

```text
                       split system_id  n_trajectories  horizon  state_dim  action_dim  disturbance_dim  action_min  action_max  disturbance_0_max                   scenario_type  disturbance_1_max
                       train      cstr             180       50          2           1                3    8.141234   11.200000           1.120000                   normal_policy         346.000000
                     id_test      cstr              40       50          2           1                3    8.239336   11.106145           1.110516                   normal_policy         345.964128
        ood_action_magnitude      cstr              40       50          2           1                3   11.800000   17.087133           1.120000             cooling_step_change         346.000000
            ood_inflow_spike      cstr              40       50          2           1                3    8.285027   11.152559           1.677311        feed_concentration_spike         345.416525
                ood_combined      cstr              40       50          2           1                3    4.735064   10.934555           1.545168 combined_feed_and_cooling_shift         367.787587
  ood_feed_temperature_spike      cstr              40       50          2           1                3    8.111571   11.043029           1.101002          feed_temperature_spike         380.000000
     ood_reaction_rate_shift      cstr              40       50          2           1                3    8.258708   11.043599           1.101670             reaction_rate_shift         346.000000
ood_unsafe_temperature_event      cstr              40       50          2           1                3    3.000000   10.942022           1.116100        unsafe_temperature_event         392.000000
```

## Model summary

```text
       model_id                        split  rmse_mean  mae_mean  max_abs_error_mean
      hold_last                      id_test   2.745613  1.865023            5.554444
      hold_last         ood_action_magnitude   2.251354  1.439311            5.502094
      hold_last                 ood_combined   5.557372  3.329628           13.719080
      hold_last   ood_feed_temperature_spike   3.123327  2.072850            6.233200
      hold_last             ood_inflow_spike   3.393987  2.195045            7.245571
      hold_last      ood_reaction_rate_shift   2.724642  1.823345            5.724832
      hold_last ood_unsafe_temperature_event  11.021695  6.286865           28.982466
    linear_narx                      id_test   0.051844  0.034024            0.110800
    linear_narx         ood_action_magnitude   0.056604  0.034284            0.146403
    linear_narx                 ood_combined   0.160650  0.084115            0.566460
    linear_narx   ood_feed_temperature_spike   0.060374  0.038700            0.129503
    linear_narx             ood_inflow_spike   0.093710  0.060487            0.188694
    linear_narx      ood_reaction_rate_shift   0.166093  0.083158            0.579744
    linear_narx ood_unsafe_temperature_event   1.202759  0.677096            2.958973
mlp_state_space                      id_test   0.003841  0.002486            0.009280
mlp_state_space         ood_action_magnitude   2.538080  1.609910            5.323228
mlp_state_space                 ood_combined   1.491625  0.811491            3.604526
mlp_state_space   ood_feed_temperature_spike   0.259714  0.147433            0.524495
mlp_state_space             ood_inflow_spike   0.404023  0.217318            1.027224
mlp_state_space      ood_reaction_rate_shift   0.293547  0.145179            0.966171
```

## In-distribution performance

```text
       model_id  rmse_mean  mae_mean  final_state_error_mean
      hold_last   2.745613  1.865023                5.561303
    linear_narx   0.051844  0.034024                0.085098
mlp_state_space   0.003841  0.002486                0.007160
```

## OOD performance

```text
       model_id                        split  rmse_mean  mae_mean  final_state_error_mean
      hold_last         ood_action_magnitude   2.251354  1.439311                5.510914
      hold_last                 ood_combined   5.557372  3.329628               13.721353
      hold_last   ood_feed_temperature_spike   3.123327  2.072850                6.237839
      hold_last             ood_inflow_spike   3.393987  2.195045                7.251030
      hold_last      ood_reaction_rate_shift   2.724642  1.823345                5.725477
      hold_last ood_unsafe_temperature_event  11.021695  6.286865               28.983672
    linear_narx         ood_action_magnitude   0.056604  0.034284                0.117664
    linear_narx                 ood_combined   0.160650  0.084115                0.566324
    linear_narx   ood_feed_temperature_spike   0.060374  0.038700                0.107712
    linear_narx             ood_inflow_spike   0.093710  0.060487                0.149466
    linear_narx      ood_reaction_rate_shift   0.166093  0.083158                0.564950
    linear_narx ood_unsafe_temperature_event   1.202759  0.677096                2.959029
mlp_state_space         ood_action_magnitude   2.538080  1.609910                5.323462
mlp_state_space                 ood_combined   1.491625  0.811491                3.604557
mlp_state_space   ood_feed_temperature_spike   0.259714  0.147433                0.524448
mlp_state_space             ood_inflow_spike   0.404023  0.217318                1.027267
mlp_state_space      ood_reaction_rate_shift   0.293547  0.145179                0.968997
mlp_state_space ood_unsafe_temperature_event   7.246937  3.900641               19.674524
```

## Risk-coverage summary

risk_coverage_rows: 1008
scenario_score_rows: 840

```text
         judge_id  coverage  false_accept_rate
  combined_linear       0.1            0.52381
disagreement_only       0.1            0.52381
   invariant_only       0.1            0.52381
oracle_error_rank       0.1            0.52381
  random_baseline       0.1            0.52381
      repair_only       0.1            0.52381
     support_only       0.1            0.52381
 uncertainty_only       0.1            0.52381
  combined_linear       0.2            0.52381
disagreement_only       0.2            0.52381
   invariant_only       0.2            0.52381
oracle_error_rank       0.2            0.52381
  random_baseline       0.2            0.52381
      repair_only       0.2            0.52381
     support_only       0.2            0.52381
 uncertainty_only       0.2            0.52381
  combined_linear       0.4            0.52381
disagreement_only       0.4            0.52381
   invariant_only       0.4            0.52381
oracle_error_rank       0.4            0.52381
```

## Best judge by coverage

```text
       judge_id  coverage  false_accept_rate
combined_linear       0.1            0.52381
combined_linear       0.2            0.52381
combined_linear       0.4            0.52381
combined_linear       0.6            0.52381
combined_linear       0.8            0.52381
combined_linear       1.0            0.52381
```

## Did combined judge beat simple judges?

Combined_linear matched or beat the strongest single-signal judge at every configured coverage in this smoke run.

## Claim status

Result: NOT EVALUATED

Explanation:
This run did not include the v1 multi-system claim-status evaluation.

## Known failures

- none

## Reproduction command

```bash
python scripts/run_smoke.py --config configs/experiments/smoke_cstr.yaml
```
