# Failure Table Report

## Input files

- results/smoke_two_tank/risk_coverage.csv
- results/smoke_two_tank/scenario_scores.csv
- results/smoke_two_tank/model_metrics.csv
- results/smoke_two_tank/summary.json

## Output files

- results/failure_analysis/failure_table.csv
- results/failure_analysis/failure_table_schema.json

## Row count

28800

## Scenario count

600

## Models

hold_last, linear_narx, mlp_state_space

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
