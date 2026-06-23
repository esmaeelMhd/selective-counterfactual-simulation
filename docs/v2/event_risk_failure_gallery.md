# Event-Risk Failure Gallery

## Example 1: accepted event-risk false accept

- system_id: cstr
- model_id: hold_last
- scenario_id: cstr_seed1_judge_test_unsafe_temperature_event_0008
- scenario_type: unsafe_temperature_event
- badness_target: bad_event
- judge_id: calibration_selected_candidate_ranker
- coverage: 0.05
- accepted/refused: accepted
- false_accept: True
- rmse: 3.780781
- event_bad: True
- risk_score: -1.398566
- baseline_judge: 
- source_row_id: 39490
- Trajectory plot unavailable from current artifacts.

## Example 2: RMSE-acceptable but event-bad case

- system_id: cstr
- model_id: gradient_boosted_narx
- scenario_id: cstr_seed0_judge_test_id_0003
- scenario_type: id
- badness_target: bad_event
- judge_id: calibration_selected_candidate_ranker
- coverage: 0.05
- accepted/refused: refused
- false_accept: False
- rmse: 0.029507
- event_bad: True
- risk_score: 0.000000
- baseline_judge: 
- source_row_id: 34619
- Trajectory plot unavailable from current artifacts.

## Example 3: calibrated judge loses to fair deployable baseline

- system_id: cstr
- model_id: gradient_boosted_narx
- scenario_id: source_row_9448
- scenario_type: unknown
- badness_target: bad_event
- judge_id: calibration_selected_candidate_ranker
- coverage: 0.1
- accepted/refused: accepted
- false_accept: True
- rmse: 0.142857
- event_bad: True
- risk_score: 0.142857
- baseline_judge: conformal_risk_threshold
- source_row_id: 9448
- Trajectory plot unavailable from current artifacts.

## Example 4: row-wise envelope illustrates diagnostic ceiling

- system_id: cstr
- model_id: gradient_boosted_narx
- scenario_id: source_row_6840
- scenario_type: unknown
- badness_target: bad_event
- judge_id: calibration_selected_candidate_ranker
- coverage: 0.05
- accepted/refused: accepted
- false_accept: True
- rmse: 0.250000
- event_bad: True
- risk_score: 0.250000
- baseline_judge: invariant_only
- source_row_id: 6840
- Trajectory plot unavailable from current artifacts.

## Example 5: correctly refused event-risk case

- system_id: cstr
- model_id: gradient_boosted_narx
- scenario_id: cstr_seed0_judge_test_id_0003
- scenario_type: id
- badness_target: bad_event
- judge_id: calibration_selected_candidate_ranker
- coverage: 0.05
- accepted/refused: refused
- false_accept: False
- rmse: 0.029507
- event_bad: True
- risk_score: 0.000000
- baseline_judge: 
- source_row_id: 34619
- Trajectory plot unavailable from current artifacts.
