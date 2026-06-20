# TwoTank Severity Sweep Report

## Command

```bash
python scripts/run_severity_sweep.py --config /tmp/pytest-of-ismayil/pytest-96/test_severity_sweep_tiny_run0/tiny.yaml --severities low high --output /tmp/pytest-of-ismayil/pytest-96/test_severity_sweep_tiny_run0/severity
```

## Severity definitions

See `docs/two_tank_intervention_severity.md`.

## Error vs severity

| severity | mean_error | std_error |
| --- | ---: | ---: |
| low | 0.151620 | 0.177097 |
| high | 0.281814 | 0.207617 |

## Validator signals vs severity

| severity | support | uncertainty | disagreement | invariant | repair |
| --- | ---: | ---: | ---: | ---: | ---: |
| low | 5.229192 | 0.002495 | 0.247301 | 0.005754 | 0.000000 |
| high | 18.760376 | 0.002517 | 0.405839 | 0.024174 | 0.000000 |

## False accept rate vs severity

| severity | judge_id | coverage | false_accept_rate |
| --- | --- | ---: | ---: |
| high | combined_linear | 0.100000 | 0.533333 |
| high | combined_linear | 0.200000 | 0.600000 |
| high | combined_linear | 0.400000 | 0.622222 |
| high | combined_linear | 0.600000 | 0.616667 |
| high | combined_linear | 0.800000 | 0.613333 |
| high | combined_linear | 1.000000 | 0.622222 |
| high | disagreement_only | 0.100000 | 0.533333 |
| high | disagreement_only | 0.200000 | 0.600000 |
| high | disagreement_only | 0.400000 | 0.644444 |
| high | disagreement_only | 0.600000 | 0.633333 |
| high | disagreement_only | 0.800000 | 0.626667 |
| high | disagreement_only | 1.000000 | 0.622222 |
| high | invariant_only | 0.100000 | 0.666667 |
| high | invariant_only | 0.200000 | 0.666667 |
| high | invariant_only | 0.400000 | 0.666667 |
| high | invariant_only | 0.600000 | 0.650000 |
| high | invariant_only | 0.800000 | 0.613333 |
| high | invariant_only | 1.000000 | 0.622222 |
| high | oracle_error_rank | 0.100000 | 0.466667 |
| high | oracle_error_rank | 0.200000 | 0.533333 |
| high | oracle_error_rank | 0.400000 | 0.555556 |
| high | oracle_error_rank | 0.600000 | 0.583333 |
| high | oracle_error_rank | 0.800000 | 0.600000 |
| high | oracle_error_rank | 1.000000 | 0.622222 |
| high | random_baseline | 0.100000 | 0.600000 |
| high | random_baseline | 0.200000 | 0.566667 |
| high | random_baseline | 0.400000 | 0.600000 |
| high | random_baseline | 0.600000 | 0.600000 |
| high | random_baseline | 0.800000 | 0.600000 |
| high | random_baseline | 1.000000 | 0.622222 |
| high | repair_only | 0.100000 | 0.666667 |
| high | repair_only | 0.200000 | 0.666667 |
| high | repair_only | 0.400000 | 0.622222 |
| high | repair_only | 0.600000 | 0.633333 |
| high | repair_only | 0.800000 | 0.626667 |
| high | repair_only | 1.000000 | 0.622222 |
| high | support_only | 0.100000 | 0.600000 |
| high | support_only | 0.200000 | 0.600000 |
| high | support_only | 0.400000 | 0.622222 |
| high | support_only | 0.600000 | 0.583333 |
| high | support_only | 0.800000 | 0.600000 |
| high | support_only | 1.000000 | 0.622222 |
| high | uncertainty_only | 0.100000 | 0.666667 |
| high | uncertainty_only | 0.200000 | 0.666667 |
| high | uncertainty_only | 0.400000 | 0.666667 |
| high | uncertainty_only | 0.600000 | 0.633333 |
| high | uncertainty_only | 0.800000 | 0.626667 |
| high | uncertainty_only | 1.000000 | 0.622222 |
| low | combined_linear | 0.100000 | 0.000000 |
| low | combined_linear | 0.200000 | 0.166667 |
| low | combined_linear | 0.400000 | 0.200000 |
| low | combined_linear | 0.600000 | 0.233333 |
| low | combined_linear | 0.800000 | 0.266667 |
| low | combined_linear | 1.000000 | 0.288889 |
| low | disagreement_only | 0.100000 | 0.133333 |
| low | disagreement_only | 0.200000 | 0.166667 |
| low | disagreement_only | 0.400000 | 0.200000 |
| low | disagreement_only | 0.600000 | 0.233333 |
| low | disagreement_only | 0.800000 | 0.266667 |
| low | disagreement_only | 1.000000 | 0.288889 |
| low | invariant_only | 0.100000 | 0.200000 |
| low | invariant_only | 0.200000 | 0.200000 |
| low | invariant_only | 0.400000 | 0.266667 |
| low | invariant_only | 0.600000 | 0.283333 |
| low | invariant_only | 0.800000 | 0.266667 |
| low | invariant_only | 1.000000 | 0.288889 |
| low | oracle_error_rank | 0.100000 | 0.000000 |
| low | oracle_error_rank | 0.200000 | 0.100000 |
| low | oracle_error_rank | 0.400000 | 0.155556 |
| low | oracle_error_rank | 0.600000 | 0.216667 |
| low | oracle_error_rank | 0.800000 | 0.253333 |
| low | oracle_error_rank | 1.000000 | 0.288889 |
| low | random_baseline | 0.100000 | 0.266667 |
| low | random_baseline | 0.200000 | 0.266667 |
| low | random_baseline | 0.400000 | 0.266667 |
| low | random_baseline | 0.600000 | 0.300000 |
| low | random_baseline | 0.800000 | 0.293333 |
| low | random_baseline | 1.000000 | 0.288889 |
| low | repair_only | 0.100000 | 0.266667 |
| low | repair_only | 0.200000 | 0.333333 |

## Monotonicity checks

Did error increase with severity? True
Did support distance increase with severity? True
Did false accept rate change with severity? True

## Verdict

MEANINGFUL

## Explanation

Error increased and these validator signals increased: mean_support_distance, mean_disagreement_score, mean_invariant_residual.

## Known failures

- none
