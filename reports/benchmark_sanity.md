# Benchmark Sanity Report

## OOD distribution checks

| check | passed | value | threshold |
| --- | ---: | ---: | ---: |
| ood_action_magnitude_action_range_ratio | True | 2.363897 | 1.250000 |
| ood_action_magnitude_disturbance0_max_ratio | False | 0.975155 | 1.250000 |
| ood_inflow_spike_action_range_ratio | False | 0.888760 | 1.250000 |
| ood_inflow_spike_disturbance0_max_ratio | True | 2.764866 | 1.250000 |
| ood_combined_action_range_ratio | True | 3.234316 | 1.250000 |
| ood_combined_disturbance0_max_ratio | True | 2.450301 | 1.250000 |

## Error separation

| split | mean_error | bad_rate |
| --- | ---: | ---: |
| id_test | 0.171247 | 0.300000 |
| ood_action_magnitude | 0.420843 | 0.966667 |
| ood_combined | 0.474653 | 1.000000 |
| ood_inflow_spike | 0.270846 | 0.666667 |

## Label degeneracy

Bad label rate: 0.733333; non-degenerate: True; enough bad scenarios: True; enough accepted bad scenarios: True.

## Event degeneracy

Event labels available: False. Current v0 artifacts do not store raw event trajectories.

## Benchmark verdict

VALID_BENCHMARK

## Explanation

OOD mean error 0.388781 vs ID mean error 0.171247.

## Required fixes if weak or invalid

- none
