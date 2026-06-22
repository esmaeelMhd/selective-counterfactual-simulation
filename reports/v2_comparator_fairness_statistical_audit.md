# v2 Comparator Fairness Statistical Audit

## Verdict

CALIBRATED_TARGET_DEPENDENT

## Fair Deployable Baseline Effect

| system_id | badness_target | mean_far_margin | bootstrap_ci_low | bootstrap_ci_high | seed_win_rate | practical_threshold_pass |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| cstr | bad_event | 0.001071 | -0.017500 | 0.019643 | 0.300000 | False |
| cstr | bad_rmse | 0.000238 | 0.000000 | 0.000714 | 0.100000 | False |
| cstr | bad_rmse_or_event | 0.000238 | 0.000000 | 0.000714 | 0.100000 | False |
| heat_exchanger | bad_event | -0.011786 | -0.036795 | 0.010714 | 0.200000 | False |
| heat_exchanger | bad_rmse | 0.000000 | 0.000000 | 0.000000 | 0.000000 | False |
| heat_exchanger | bad_rmse_or_event | 0.001012 | 0.000000 | 0.003036 | 0.100000 | False |
| two_tank | bad_event | 0.000000 | 0.000000 | 0.000000 | 0.000000 | False |
| two_tank | bad_rmse | 0.000556 | -0.019561 | 0.024450 | 0.300000 | False |
| two_tank | bad_rmse_or_event | -0.001667 | -0.021675 | 0.024117 | 0.200000 | False |

## Diagnostic Envelope Effect

| system_id | badness_target | mean_far_margin | bootstrap_ci_low | bootstrap_ci_high | seed_win_rate |
| --- | --- | ---: | ---: | ---: | ---: |
| cstr | bad_event | -0.045357 | -0.078589 | -0.020705 | 0.000000 |
| cstr | bad_rmse | -0.001310 | -0.003274 | 0.000000 | 0.000000 |
| cstr | bad_rmse_or_event | -0.001726 | -0.003690 | -0.000417 | 0.000000 |
| heat_exchanger | bad_event | -0.063214 | -0.085000 | -0.041429 | 0.000000 |
| heat_exchanger | bad_rmse | -0.001964 | -0.003214 | -0.000833 | 0.000000 |
| heat_exchanger | bad_rmse_or_event | -0.004643 | -0.008393 | -0.001667 | 0.000000 |
| two_tank | bad_event | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| two_tank | bad_rmse | -0.043778 | -0.053444 | -0.033442 | 0.000000 |
| two_tank | bad_rmse_or_event | -0.043778 | -0.053556 | -0.033661 | 0.000000 |

## RMSE Target Result

{'mean_margin': 0.00026455026455026435, 'positive_system_count': 2}

## Event-Risk Target Result

{'mean_margin': -0.0035714285714285713, 'event_risk_worsening_count': 1}
