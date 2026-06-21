# Event-Risk Audit

## Question

Does calibrated low-coverage refusal reduce event-risk false accepts, not only RMSE false accepts?

## Event definitions

{
  "cstr": {
    "event_labels": [
      "temperature_above_limit",
      "concentration_out_of_safe_range",
      "unsafe_reactor_state"
    ],
    "thresholds": {
      "concentration_high": 1.8,
      "concentration_low": 0.25,
      "temperature_high": 390.0
    }
  },
  "two_tank": {
    "event_labels": [
      "overflow_event",
      "underflow_event"
    ],
    "thresholds": {
      "level_max": 10.0,
      "level_min": 0.0
    }
  }
}

## Event-label availability

| system | event_labels_available | missing_reason |
| --- | ---: | --- |
| two_tank | True | none |
| cstr | True | none |

## Bad RMSE vs bad event comparison

| system | coverage | judge | rmse_far | event_far | rmse_or_event_far |
| --- | ---: | --- | ---: | ---: | ---: |
| two_tank | 0.050000 | best_single_signal_selected_on_calibration | 0.640000 | 0.000000 | 0.640000 |
| two_tank | 0.050000 | calibration_selected_candidate_ranker | 0.466667 | 0.000000 | 0.466667 |
| two_tank | 0.100000 | best_single_signal_selected_on_calibration | 0.653333 | 0.000000 | 0.653333 |
| two_tank | 0.100000 | rank_normalized_linear | 0.573333 | 0.000000 | 0.573333 |
| cstr | 0.050000 | best_single_signal_selected_on_calibration | 0.666667 | 0.000000 | 0.666667 |
| cstr | 0.050000 | rank_normalized_linear | 0.666667 | 0.000000 | 0.666667 |
| cstr | 0.100000 | best_single_signal_selected_on_calibration | 0.666667 | 0.000000 | 0.666667 |
| cstr | 0.100000 | rank_normalized_linear | 0.666667 | 0.000000 | 0.666667 |

## CSTR event-risk result

- coverage 0.05, judge best_single_signal_selected_on_calibration: event FAR 0.000000, RMSE-or-event FAR 0.666667
- coverage 0.05, judge rank_normalized_linear: event FAR 0.000000, RMSE-or-event FAR 0.666667
- coverage 0.1, judge best_single_signal_selected_on_calibration: event FAR 0.000000, RMSE-or-event FAR 0.666667
- coverage 0.1, judge rank_normalized_linear: event FAR 0.000000, RMSE-or-event FAR 0.666667

## TwoTank event-risk result

- coverage 0.05, judge best_single_signal_selected_on_calibration: event FAR 0.000000, RMSE-or-event FAR 0.640000
- coverage 0.05, judge calibration_selected_candidate_ranker: event FAR 0.000000, RMSE-or-event FAR 0.466667
- coverage 0.1, judge best_single_signal_selected_on_calibration: event FAR 0.000000, RMSE-or-event FAR 0.653333
- coverage 0.1, judge rank_normalized_linear: event FAR 0.000000, RMSE-or-event FAR 0.573333

## Verdict

EVENT_SUPPORTS_CLAIM

## Explanation

Event labels were computed from true and predicted trajectories using explicit thresholds, then merged with deployable judge risk rankings.

## Known limitations

Predicted trajectories are regenerated from saved model-train/test data because earlier calibrated artifacts did not store every prediction trajectory.
