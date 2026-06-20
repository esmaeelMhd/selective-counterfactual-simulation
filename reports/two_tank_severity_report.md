# TwoTank Severity Sweep Report

## Command

```bash
python scripts/run_severity_sweep.py --config configs/experiments/smoke_two_tank.yaml --severities low medium high extreme --output results/two_tank_severity_sweep
```

## Severity definitions

See `docs/two_tank_intervention_severity.md`.

## Error vs severity

| severity | mean_error | std_error |
| --- | ---: | ---: |
| low | 0.226731 | 0.276286 |
| medium | 0.337246 | 0.264025 |
| high | 0.463089 | 0.282444 |
| extreme | 0.566311 | 0.312166 |

## Validator signals vs severity

| severity | support | uncertainty | disagreement | invariant | repair |
| --- | ---: | ---: | ---: | ---: | ---: |
| low | 4.835241 | 0.002713 | 0.404180 | 0.003459 | 0.000000 |
| medium | 9.870907 | 0.002715 | 0.522107 | 0.007339 | 0.000000 |
| high | 15.125473 | 0.002714 | 0.659031 | 0.011362 | 0.000000 |
| extreme | 19.014240 | 0.002714 | 0.763049 | 0.014913 | 0.000000 |

## False accept rate vs severity

| severity | judge_id | coverage | false_accept_rate |
| --- | --- | ---: | ---: |
| extreme | combined_linear | 0.100000 | 0.640000 |
| extreme | combined_linear | 0.200000 | 0.673333 |
| extreme | combined_linear | 0.400000 | 0.703333 |
| extreme | combined_linear | 0.600000 | 0.715556 |
| extreme | combined_linear | 0.800000 | 0.730000 |
| extreme | combined_linear | 1.000000 | 0.738667 |
| extreme | disagreement_only | 0.100000 | 0.613333 |
| extreme | disagreement_only | 0.200000 | 0.673333 |
| extreme | disagreement_only | 0.400000 | 0.703333 |
| extreme | disagreement_only | 0.600000 | 0.713333 |
| extreme | disagreement_only | 0.800000 | 0.730000 |
| extreme | disagreement_only | 1.000000 | 0.738667 |
| extreme | invariant_only | 0.100000 | 0.773333 |
| extreme | invariant_only | 0.200000 | 0.766667 |
| extreme | invariant_only | 0.400000 | 0.750000 |
| extreme | invariant_only | 0.600000 | 0.746667 |
| extreme | invariant_only | 0.800000 | 0.740000 |
| extreme | invariant_only | 1.000000 | 0.738667 |
| extreme | oracle_error_rank | 0.100000 | 0.613333 |
| extreme | oracle_error_rank | 0.200000 | 0.673333 |
| extreme | oracle_error_rank | 0.400000 | 0.703333 |
| extreme | oracle_error_rank | 0.600000 | 0.713333 |
| extreme | oracle_error_rank | 0.800000 | 0.723333 |
| extreme | oracle_error_rank | 1.000000 | 0.738667 |
| extreme | random_baseline | 0.100000 | 0.706667 |
| extreme | random_baseline | 0.200000 | 0.726667 |
| extreme | random_baseline | 0.400000 | 0.740000 |
| extreme | random_baseline | 0.600000 | 0.737778 |
| extreme | random_baseline | 0.800000 | 0.736667 |
| extreme | random_baseline | 1.000000 | 0.738667 |
| extreme | repair_only | 0.100000 | 0.733333 |
| extreme | repair_only | 0.200000 | 0.746667 |
| extreme | repair_only | 0.400000 | 0.743333 |
| extreme | repair_only | 0.600000 | 0.744444 |
| extreme | repair_only | 0.800000 | 0.736667 |
| extreme | repair_only | 1.000000 | 0.738667 |
| extreme | support_only | 0.100000 | 0.706667 |
| extreme | support_only | 0.200000 | 0.713333 |
| extreme | support_only | 0.400000 | 0.720000 |
| extreme | support_only | 0.600000 | 0.724444 |
| extreme | support_only | 0.800000 | 0.730000 |
| extreme | support_only | 1.000000 | 0.738667 |
| extreme | uncertainty_only | 0.100000 | 0.720000 |
| extreme | uncertainty_only | 0.200000 | 0.726667 |
| extreme | uncertainty_only | 0.400000 | 0.736667 |
| extreme | uncertainty_only | 0.600000 | 0.740000 |
| extreme | uncertainty_only | 0.800000 | 0.740000 |
| extreme | uncertainty_only | 1.000000 | 0.738667 |
| high | combined_linear | 0.100000 | 0.640000 |
| high | combined_linear | 0.200000 | 0.673333 |
| high | combined_linear | 0.400000 | 0.703333 |
| high | combined_linear | 0.600000 | 0.713333 |
| high | combined_linear | 0.800000 | 0.718333 |
| high | combined_linear | 1.000000 | 0.721333 |
| high | disagreement_only | 0.100000 | 0.613333 |
| high | disagreement_only | 0.200000 | 0.673333 |
| high | disagreement_only | 0.400000 | 0.703333 |
| high | disagreement_only | 0.600000 | 0.713333 |
| high | disagreement_only | 0.800000 | 0.718333 |
| high | disagreement_only | 1.000000 | 0.721333 |
| high | invariant_only | 0.100000 | 0.720000 |
| high | invariant_only | 0.200000 | 0.726667 |
| high | invariant_only | 0.400000 | 0.720000 |
| high | invariant_only | 0.600000 | 0.722222 |
| high | invariant_only | 0.800000 | 0.720000 |
| high | invariant_only | 1.000000 | 0.721333 |
| high | oracle_error_rank | 0.100000 | 0.613333 |
| high | oracle_error_rank | 0.200000 | 0.673333 |
| high | oracle_error_rank | 0.400000 | 0.703333 |
| high | oracle_error_rank | 0.600000 | 0.713333 |
| high | oracle_error_rank | 0.800000 | 0.718333 |
| high | oracle_error_rank | 1.000000 | 0.721333 |
| high | random_baseline | 0.100000 | 0.693333 |
| high | random_baseline | 0.200000 | 0.713333 |
| high | random_baseline | 0.400000 | 0.720000 |
| high | random_baseline | 0.600000 | 0.722222 |
| high | random_baseline | 0.800000 | 0.721667 |
| high | random_baseline | 1.000000 | 0.721333 |
| high | repair_only | 0.100000 | 0.720000 |
| high | repair_only | 0.200000 | 0.720000 |

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
