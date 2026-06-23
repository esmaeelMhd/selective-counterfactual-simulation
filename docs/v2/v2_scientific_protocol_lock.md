# v2 Scientific Protocol Lock

## Starting v1 status

The frozen v1 allowed claim remains: A weak but positive low-coverage result under the frozen protocol.

The v2 track is separate. It writes only under `configs/v2/`, `docs/v2/`, `reports/v2_*.md`, and `results/v2_scientific_strengthening/`.

## v2 research question

Under a frozen protocol, does calibrated low-coverage refusal reduce false accepts across multiple dynamical systems, model classes, seeds, thresholds, and badness targets?

## Systems

- two_tank
- cstr
- heat_exchanger

Heat exchanger must pass the v2 sanity gate before it is counted as evidence.

## Model participants

- hold_last
- linear_narx
- mlp_state_space
- ensemble_mlp
- gradient_boosted_narx

## Refusal signals

- support_distance
- uncertainty_score
- disagreement_score
- invariant_residual
- repair_amount

## Judges and baselines

Simple baselines:

- support_only
- uncertainty_only
- disagreement_only
- invariant_only
- repair_only
- random_baseline

Calibrated candidates:

- best_single_signal_selected_on_calibration
- calibration_selected_candidate_ranker
- rank_normalized_linear
- logistic_calibrated_judge
- isotonic_calibrated_judge
- quantile_rule_judge
- conservative_low_coverage_judge

Stronger baselines:

- learned_error_classifier
- conformal_risk_threshold
- ensemble_disagreement_threshold

Primary calibrated judge for v2 decision rules:

- calibration_selected_candidate_ranker

## Diagnostic-only methods

- oracle_error_rank

The oracle is diagnostic only and cannot be counted as a deployable refusal judge.

## Data splits

Each system uses separated model-training, judge-calibration, and judge-test trajectory roles.

Model training split:

- model_train

Judge calibration splits:

- judge_calibration_id
- all configured judge_calibration OOD scenario splits

Judge test splits:

- judge_test_id
- all configured judge_test OOD scenario splits

## Calibration/test separation

Simulator models fit only on `model_train`.

Calibrated judges and stronger learned baselines fit only on judge-calibration rows.

All v2 effect estimates and decision gates use only judge-test rows.

If any calibration/test scenario hash overlaps, the v2 run is invalid.

## Badness targets

- bad_rmse
- bad_event
- bad_rmse_or_event

## Event definitions

TwoTank:

- overflow_event
- underflow_event

CSTR:

- temperature_above_limit
- concentration_out_of_safe_range
- unsafe_reactor_state

Heat exchanger:

- outlet_temperature_above_limit
- outlet_temperature_below_limit
- large_temperature_tracking_error

All numeric thresholds are fixed in `configs/v2/v2_event_targets.yaml`.

## Coverage grid

- 0.01
- 0.02
- 0.05
- 0.10
- 0.20
- 0.40
- 0.60
- 0.80
- 1.00

Primary coverages:

- 0.05
- 0.10

## Threshold grid

- 0.05
- 0.10
- 0.15
- 0.20
- 0.30
- 0.50

## Seed rules

The v2 full run uses exactly these seeds:

- 0
- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9

Failed seeds are not skipped. A failed seed invalidates or fails the run.

## Statistical uncertainty rules

Report seed-level paired margins, seed win rate, bootstrap 95 percent confidence intervals, and practical threshold pass/fail.

The practical thresholds are:

- minimum absolute FAR reduction: 0.05
- minimum relative FAR reduction: 0.10
- minimum strong seed win rate: 0.70

## Decision gates

Allowed decision labels are:

- UPGRADE_TO_MODERATE_MULTI_SYSTEM_LOW_COVERAGE_CLAIM
- KEEP_WEAK_LOW_COVERAGE_BENCHMARK_CLAIM
- SYSTEM_DEPENDENT_BENCHMARK_RESULT
- NO_METHOD_CLAIM_BENCHMARK_ONLY
- INVALID_V2_PROTOCOL

The decision gate must follow numeric evidence and cannot upgrade the claim when the stated conditions are not met.

## Kill criteria

If heat_exchanger sanity fails, it cannot be used as evidence.

If v2 improves only on TwoTank but not CSTR/heat_exchanger, no stronger method claim is allowed.

If event-risk false accepts worsen, no reliability claim is allowed.

If confidence intervals overlap zero on most systems, no strong method claim is allowed.

If leakage is detected, v2 is invalid.

## Forbidden changes after results

After v2 evidence generation starts, do not change:

- judge list
- signal list
- model list
- seed list
- coverage grid
- RMSE threshold grid
- badness targets
- success rules
- decision labels

Do not add a new judge after a failure. Do not remove weak systems from reports.

## Allowed claims

The v1 claim remains allowed: A weak but positive low-coverage result under the frozen protocol.

A v2 claim may only be stated if `reports/v2_scientific_decision_gate.md` permits it.

## Forbidden claims

- safety certification
- trusted simulator
- validated digital twin
- industrial AI breakthrough
- general simulator reliability
- plant-wide validity
- high-coverage reliability
- product readiness
