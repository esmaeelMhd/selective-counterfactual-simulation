# Calibrated Judge Threshold/Coverage Stress Report

## Thresholds tested

0.05, 0.1, 0.15, 0.2, 0.3, 0.5

## Coverages tested

0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0

## Seeds tested

0, 1, 2, 3, 4

## Result by threshold

| threshold | low_coverage_win_rate | verdict |
| ---: | ---: | --- |
| 0.050000 | 0.200000 | FAILS |
| 0.100000 | 1.000000 | WORKS |
| 0.150000 | 1.000000 | WORKS |
| 0.200000 | 1.000000 | WORKS |
| 0.300000 | 0.900000 | WORKS |
| 0.500000 | 1.000000 | WORKS |

## Result by coverage

| coverage | win_rate | mean_margin |
| ---: | ---: | ---: |
| 0.050000 | 0.833333 | 0.097778 |
| 0.100000 | 0.866667 | 0.081111 |
| 0.200000 | 0.866667 | 0.064333 |
| 0.400000 | 0.866667 | 0.038278 |
| 0.600000 | 0.833333 | 0.019593 |
| 0.800000 | 0.833333 | 0.007417 |
| 1.000000 | 0.000000 | -0.000000 |

## Regions where calibrated judge works

0.1, 0.15, 0.2, 0.3, 0.5

## Regions where calibrated judge fails

0.05

## Verdict

ROBUST_LOW_COVERAGE_ONLY
