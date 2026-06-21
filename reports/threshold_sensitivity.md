# Threshold Sensitivity Report

## Thresholds tested

0.05, 0.15, 0.5

## Bad scenario rate by threshold

| threshold | bad_rate |
| ---: | ---: |
| 0.050000 | 1.000000 |
| 0.150000 | 0.583333 |
| 0.500000 | 0.000000 |

## Combined judge result by threshold

| threshold | combined_win_rate | strongest_simple_judge | verdict |
| ---: | ---: | --- | --- |
| 0.050000 | 0.000000 | UNAVAILABLE | UNAVAILABLE |
| 0.150000 | 0.000000 | disagreement_only | NOT_SUPPORTED |
| 0.500000 | 0.000000 | UNAVAILABLE | UNAVAILABLE |

## Thresholds where combined works

none

## Thresholds where combined fails

0.05, 0.15, 0.5

## Interpretation

combined_linear worked at 0 of 3 thresholds. Thresholds with degenerate labels are marked unavailable, not counted as wins.

## Verdict

UNSUPPORTED_ACROSS_THRESHOLDS
