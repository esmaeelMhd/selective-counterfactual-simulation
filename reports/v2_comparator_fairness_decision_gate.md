# v2 Comparator Fairness Decision Gate

## Starting v2 decision

NO_METHOD_CLAIM_BENCHMARK_ONLY

## Comparator taxonomy

Row-wise strongest-baseline envelope is diagnostic only. Deployable baselines are selected from calibration rows only.

## Selection validity

Selection verdict: COMPARATOR_SELECTION_VALID

Deployable selection uses test labels: False

## Fair comparator results

Fair mode: per_system_target_calibration_selected_baseline

Fair mean margin: -0.0011485890652557322

Positive systems: ['cstr']

## Diagnostic envelope results

Row-wise mode: row_wise_strongest_baseline_envelope

Row-wise mean margin: -0.022863315696649025

## RMSE vs event-risk

RMSE target result: {'mean_margin': 0.00026455026455026435, 'positive_system_count': 2}

Event target result: {'event_risk_worsening_count': 1, 'mean_margin': -0.0035714285714285713}

## Decision

CALIBRATED_TARGET_DEPENDENT

## Allowed claim

Calibrated refusal is target-dependent and not reliable for event-risk.

## Forbidden claims

- row-wise strongest-baseline envelope is deployable
- calibrated refusal works generally
- event-risk refusal is reliable
- safety certification
- trusted simulator
- product-ready digital twin

## Recommended next action

Do not expand systems or claims until the event-risk failure mode and fair-baseline result are understood.
