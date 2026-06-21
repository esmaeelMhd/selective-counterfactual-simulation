# Benchmark Sanity Report

## OOD distribution checks

| check | passed | value | threshold |
| --- | ---: | ---: | ---: |
| ood_action_magnitude_action_range_ratio | True | 3.000000 | 1.250000 |
| ood_action_magnitude_disturbance0_max_ratio | False | 1.000000 | 1.250000 |
| ood_inflow_spike_action_range_ratio | False | 1.000000 | 1.250000 |
| ood_inflow_spike_disturbance0_max_ratio | True | 3.200000 | 1.250000 |
| ood_combined_action_range_ratio | True | 3.000000 | 1.250000 |
| ood_combined_disturbance0_max_ratio | True | 3.300000 | 1.250000 |

## Error separation

| split | mean_error | bad_rate |
| --- | ---: | ---: |
| id_test | 1.000000 | 1.000000 |
| ood_action_magnitude | 1.000000 | 1.000000 |
| ood_combined | 1.000000 | 1.000000 |
| ood_inflow_spike | 1.000000 | 1.000000 |

## Label degeneracy

Bad label rate: 1.000000; non-degenerate: False; enough bad scenarios: True; enough accepted bad scenarios: True.

## Event degeneracy

Event labels available: False. Current v0 artifacts do not store raw event trajectories.

## Benchmark verdict

WEAK_BENCHMARK

## Explanation

OOD mean error 1.000000 vs ID mean error 1.000000.

## Required fixes if weak or invalid

- Improve error separation and store event trajectories for event-label analysis.
