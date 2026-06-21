# CSTR Sanity Report

## Config

experiment_id: calibrated_cstr_tiny
seed: 11
horizon: 12
dt: 0.1

## CSTR dynamics summary

state: concentration and temperature. action: cooling command. disturbances: feed concentration, feed temperature, and flow-rate/reaction shift proxy.

## Data split summary

Finite: True
Nonconstant: True
Physically plausible: True
Scenario types: combined_feed_and_cooling_shift, cooling_step_change, feed_concentration_spike, feed_temperature_spike, id, reaction_rate_shift, unsafe_temperature_event

## Distribution checks

| check | value | threshold | passed |
| --- | ---: | ---: | ---: |
| cooling_action_shift | 2.597184 | 1.250000 | True |
| feed_concentration_shift | 1.544343 | 1.250000 | True |
| feed_temperature_shift | 32.215984 | 10.000000 | True |

## Model error checks

| model | id_rmse | ood_rmse | ood_minus_id | passed |
| --- | ---: | ---: | ---: | ---: |
| hold_last | 0.991790 | 1.426657 | 0.434867 | True |
| linear_narx | 0.025054 | 0.124615 | 0.099561 | True |

## Label checks

| role | row_count | bad_count | bad_rate | non_degenerate |
| --- | ---: | ---: | ---: | ---: |
| judge_calibration | 112 | 61 | 0.544643 | True |
| judge_test | 140 | 79 | 0.564286 | True |

## Split overlap checks

scenario_overlap_count: 0

## Verdict

VALID_CSTR_BENCHMARK

## Explanation

Invalid reasons: none
Weak reasons: none

## Required fixes if weak or invalid

none
