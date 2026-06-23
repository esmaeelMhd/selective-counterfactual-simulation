# v2 Frozen Protocol Report

## Verdict

V2_FROZEN_PROTOCOL_COMPLETE

## Valid Systems

two_tank, cstr

## Models

hold_last, linear_narx

## Badness Targets

bad_rmse, bad_event, bad_rmse_or_event

## Risk Coverage Preview

| system_id | seed | model_id | badness_target | judge_id | coverage | false_accept_rate | baseline_judge | absolute_margin |
| --- | ---: | --- | --- | --- | ---: | ---: | --- | ---: |
| two_tank | 0 | hold_last | bad_rmse | support_only | 0.010000 | 1.000000 | conformal_risk_threshold | 0.000000 |
| two_tank | 0 | hold_last | bad_rmse | support_only | 0.020000 | 1.000000 | conformal_risk_threshold | 0.000000 |
| two_tank | 0 | hold_last | bad_rmse | support_only | 0.050000 | 1.000000 | conformal_risk_threshold | 0.000000 |
| two_tank | 0 | hold_last | bad_rmse | support_only | 0.100000 | 1.000000 | conformal_risk_threshold | 0.000000 |
| two_tank | 0 | hold_last | bad_rmse | support_only | 0.200000 | 1.000000 | conformal_risk_threshold | 0.000000 |
| two_tank | 0 | hold_last | bad_rmse | uncertainty_only | 0.010000 | 1.000000 | conformal_risk_threshold | 0.000000 |
| two_tank | 0 | hold_last | bad_rmse | uncertainty_only | 0.020000 | 1.000000 | conformal_risk_threshold | 0.000000 |
| two_tank | 0 | hold_last | bad_rmse | uncertainty_only | 0.050000 | 1.000000 | conformal_risk_threshold | 0.000000 |
| two_tank | 0 | hold_last | bad_rmse | uncertainty_only | 0.100000 | 1.000000 | conformal_risk_threshold | 0.000000 |
| two_tank | 0 | hold_last | bad_rmse | uncertainty_only | 0.200000 | 1.000000 | conformal_risk_threshold | 0.000000 |
| two_tank | 0 | hold_last | bad_rmse | disagreement_only | 0.010000 | 1.000000 | conformal_risk_threshold | 0.000000 |
| two_tank | 0 | hold_last | bad_rmse | disagreement_only | 0.020000 | 1.000000 | conformal_risk_threshold | 0.000000 |

## Leakage

Detected: False
