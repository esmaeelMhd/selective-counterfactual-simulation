# Failure Gallery

## Why this gallery exists

These examples make the benchmark concrete. They are selected from actual calibrated test-table artifacts, not fabricated cases.

## Example 1: Accepted good scenario

System: two_tank

Model: mlp_state_space

Scenario type: normal_policy

Judge: calibration_selected_candidate_ranker

Coverage: 0.05

RMSE: 0.000451

Decision: accepted

False accept status: False

Key signal values: support=1.436750, uncertainty=0.001577, disagreement=0.071492, invariant=0.000101, repair=0.000000, risk=0.110857

Source artifact: results/calibrated_two_tank/test_table.csv

![Example 1: Accepted good scenario](docs/figures/failure_gallery/example_1_accepted_good.png)


## Example 2: Correctly rejected bad scenario

System: two_tank

Model: hold_last

Scenario type: held_out_action_magnitude

Judge: calibration_selected_candidate_ranker

Coverage: 0.05

RMSE: 1.380867

Decision: refused

False accept status: False

Key signal values: support=15.219363, uncertainty=0.000000, disagreement=1.132658, invariant=0.005934, repair=0.000000, risk=1.132658

Source artifact: results/calibrated_two_tank/test_table.csv

![Example 2: Correctly rejected bad scenario](docs/figures/failure_gallery/example_2_correctly_rejected_bad.png)


## Example 3: False accept

System: cstr

Model: hold_last

Scenario type: id

Judge: support_only

Coverage: 0.05

RMSE: 3.785052

Decision: accepted

False accept status: True

Key signal values: support=1.766893, uncertainty=0.000000, disagreement=2.592008, invariant=0.359716, repair=0.000000, risk=1.766893

Source artifact: results/calibrated_cstr/test_table.csv

![Example 3: False accept](docs/figures/failure_gallery/example_3_false_accept_cstr.png)


## Example 4: CSTR within-bound dynamic failure

System: cstr

Model: hold_last

Scenario type: unsafe_temperature_event

Judge: rank_normalized_linear

Coverage: 0.05

RMSE: 12.118251

Decision: refused

False accept status: False

Key signal values: support=26.010125, uncertainty=0.000000, disagreement=7.285496, invariant=0.810614, repair=0.000000, risk=0.995238

Source artifact: results/calibrated_cstr/test_table.csv

![Example 4: CSTR within-bound dynamic failure](docs/figures/failure_gallery/example_4_cstr_within_bound_dynamic_failure.png)


## Example 5: Invariant residual helps

System: cstr

Model: hold_last

Scenario type: unsafe_temperature_event

Judge: rank_normalized_linear

Coverage: 0.05

RMSE: 11.961385

Decision: refused

False accept status: False

Key signal values: support=25.903390, uncertainty=0.000000, disagreement=7.303224, invariant=0.850472, repair=0.000000, risk=1.000000

Source artifact: results/calibrated_cstr/test_table.csv

![Example 5: Invariant residual helps](docs/figures/failure_gallery/example_5_invariant_residual_helps.png)


## What these examples show

They show that accepted-good, correctly-rejected-bad, and false-accept cases all exist in the frozen artifacts. They also show the CSTR repair-signal blind spot: repair can be zero while invariant residual is informative.

## What these examples do not prove

They do not prove simulator safety, product readiness, broad reliability, or high-coverage performance.
