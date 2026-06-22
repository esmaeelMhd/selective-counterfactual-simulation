# TwoTank Main Result Reproduction

## Source artifact

results/calibrated_two_tank/low_coverage_summary.csv

## Result

| system_id | coverage | baseline_far | calibrated_far | absolute_margin | source_artifact | is_reproduction |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| two_tank | 0.050000 | 0.640000 | 0.466667 | 0.173333 | results/calibrated_two_tank/low_coverage_summary.csv | True |
| two_tank | 0.100000 | 0.653333 | 0.486667 | 0.166667 | results/calibrated_two_tank/low_coverage_summary.csv | True |

## Interpretation

The TwoTank low-coverage margins are nonzero: 0.173333 at coverage 0.05 and 0.166667 at coverage 0.10.

This reproduces an existing frozen artifact and does not change the current evidence manifest.

## Claim boundary

This is weak-positive low-coverage synthetic benchmark evidence only.

## Verdict

TWOTANK_MAIN_RESULT_REPRODUCED
