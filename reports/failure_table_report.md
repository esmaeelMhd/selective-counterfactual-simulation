# Failure Table Report

## Input files

- /tmp/pytest-of-ismayil/pytest-147/test_threshold_sensitivity_mar0/results/risk_coverage.csv
- /tmp/pytest-of-ismayil/pytest-147/test_threshold_sensitivity_mar0/results/scenario_scores.csv
- /tmp/pytest-of-ismayil/pytest-147/test_threshold_sensitivity_mar0/results/model_metrics.csv
- /tmp/pytest-of-ismayil/pytest-147/test_threshold_sensitivity_mar0/results/summary.json

## Output files

- /tmp/pytest-of-ismayil/pytest-147/test_threshold_sensitivity_mar0/failure_analysis/failure_table.csv
- /tmp/pytest-of-ismayil/pytest-147/test_threshold_sensitivity_mar0/failure_analysis/failure_table_schema.json

## Row count

1152

## Scenario count

48

## Models

hold_last, linear_narx

## Judges

combined_linear, disagreement_only, invariant_only, oracle_error_rank, random_baseline, repair_only, support_only, uncertainty_only

## Splits

id_test, ood_action_magnitude, ood_combined, ood_inflow_spike

## Missing columns

event_error, bad_event_label

## Numeric sanity checks

{"combined_linear_score": false, "disagreement_score": false, "invariant_residual": false, "oracle_error_rank_score": false, "random_baseline_score": false, "repair_amount": false, "support_distance": false, "uncertainty_score": false}

## Verdict

ACCEPTED
