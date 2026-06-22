# v2 Comparator Fairness Evaluation

## Verdict

FAIR_BASELINE_ANALYSIS_COMPLETE

## Deployable Comparator Check

Deployable baseline selection uses test labels: False

## Diagnostic Envelope Check

Row-wise envelope diagnostic-only: True

## RMSE and Event Summary

| comparator_mode | badness_target | mean_absolute_margin | mean_calibrated_far | mean_baseline_far | deployable | diagnostic |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| best_calibrated_family_vs_per_system_target_baseline | bad_event | -0.001905 | 0.086786 | 0.084881 | True | False |
| best_calibrated_family_vs_per_system_target_baseline | bad_rmse | 0.001185 | 0.169919 | 0.171104 | True | False |
| best_calibrated_family_vs_per_system_target_baseline | bad_rmse_or_event | -0.001519 | 0.174878 | 0.173360 | True | False |
| global_calibration_selected_baseline | bad_event | -0.001071 | 0.088452 | 0.087381 | True | False |
| global_calibration_selected_baseline | bad_rmse | 0.008598 | 0.170840 | 0.179438 | True | False |
| global_calibration_selected_baseline | bad_rmse_or_event | 0.016475 | 0.173499 | 0.189974 | True | False |
| per_system_calibration_selected_baseline | bad_event | -0.018571 | 0.088452 | 0.069881 | True | False |
| per_system_calibration_selected_baseline | bad_rmse | -0.000878 | 0.170840 | 0.169962 | True | False |
| per_system_calibration_selected_baseline | bad_rmse_or_event | -0.002657 | 0.173499 | 0.170841 | True | False |
| per_system_target_calibration_selected_baseline | bad_event | -0.003571 | 0.088452 | 0.084881 | True | False |
| per_system_target_calibration_selected_baseline | bad_rmse | 0.000265 | 0.170840 | 0.171104 | True | False |
| per_system_target_calibration_selected_baseline | bad_rmse_or_event | -0.000139 | 0.173499 | 0.173360 | True | False |
| row_wise_strongest_baseline_envelope | bad_event | -0.036190 | 0.088452 | 0.052262 | False | True |
| row_wise_strongest_baseline_envelope | bad_rmse | -0.015684 | 0.170840 | 0.155156 | False | True |
| row_wise_strongest_baseline_envelope | bad_rmse_or_event | -0.016716 | 0.173499 | 0.156783 | False | True |
