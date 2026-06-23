# v2 Comparator Fairness Summary

## Decision

CALIBRATED_TARGET_DEPENDENT

## Allowed Claim

Calibrated refusal is target-dependent and not reliable for event-risk.

## Comparator Scope

This audit compares the primary calibrated judge to deployable fixed baselines selected on calibration rows and to the diagnostic row-wise strongest-baseline envelope.

## Main Results

- fair deployable baseline mean margin: -0.0011485890652557322
- diagnostic envelope mean margin: -0.022863315696649025
- event-risk worsening count: 1
- RMSE target result: {'mean_margin': 0.00026455026455026435, 'positive_system_count': 2}
- event target result: {'event_risk_worsening_count': 1, 'mean_margin': -0.0035714285714285713}

## Interpretation

The row-wise envelope remains diagnostic only. No broad method claim is allowed unless the decision gate says so explicitly.
