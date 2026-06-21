# CSTR Diagnosis Table Report

## Input artifacts

results/calibrated_cstr/test_table.csv; results/effect_size_audit/false_accept_forensics/accepted_false_accepts.csv; results/effect_size_audit/event_risk/event_labels.csv

## Output table

results/cstr_weakness_audit/diagnosis_table/cstr_diagnosis_table.csv

## Row count

283500

## Scenario count

700

## Models

hold_last, linear_narx, mlp_state_space

## Judges

best_single_signal_selected_on_calibration, rank_normalized_linear, logistic_calibrated_judge, isotonic_calibrated_judge, quantile_rule_judge, conservative_low_coverage_judge, calibration_selected_candidate_ranker, support_only, uncertainty_only, disagreement_only, invariant_only, repair_only, combined_linear, random_baseline, oracle_error_rank

## Coverages

0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2

## Scenario types

combined_feed_and_cooling_shift, cooling_step_change, feed_concentration_spike, feed_temperature_spike, id, reaction_rate_shift, unsafe_temperature_event

## Trajectory availability

True

## Event-label availability

True

## Missing columns

none

## Verdict

ACCEPTED
