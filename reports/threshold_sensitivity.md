# Threshold Sensitivity Report

## Thresholds tested

0.05, 0.1, 0.15, 0.2, 0.3, 0.5

## Bad scenario rate by threshold

| threshold | bad_rate |
| ---: | ---: |
| 0.050000 | 0.750000 |
| 0.100000 | 0.743333 |
| 0.150000 | 0.733333 |
| 0.200000 | 0.636667 |
| 0.300000 | 0.545000 |
| 0.500000 | 0.196667 |

## Combined judge result by threshold

| threshold | combined_win_rate | strongest_simple_judge | verdict |
| ---: | ---: | --- | --- |
| 0.050000 | 0.000000 | disagreement_only | NOT_SUPPORTED |
| 0.100000 | 0.000000 | disagreement_only | NOT_SUPPORTED |
| 0.150000 | 0.000000 | disagreement_only | NOT_SUPPORTED |
| 0.200000 | 0.000000 | disagreement_only | NOT_SUPPORTED |
| 0.300000 | 0.000000 | disagreement_only | NOT_SUPPORTED |
| 0.500000 | 0.000000 | disagreement_only | NOT_SUPPORTED |

## Thresholds where combined works

none

## Thresholds where combined fails

0.05, 0.1, 0.15, 0.2, 0.3, 0.5

## Interpretation

combined_linear worked at 0 of 6 thresholds. Thresholds with degenerate labels are marked unavailable, not counted as wins.

## Verdict

UNSUPPORTED_ACROSS_THRESHOLDS
