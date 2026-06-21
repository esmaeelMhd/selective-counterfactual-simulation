# Repair vs Invariant Residual Comparison

## Question

Is repair_amount less informative than invariant_residual, especially on CSTR?

## Signal performance by system

| system | signal | auroc | auprc | spearman_rmse | near_zero_fraction |
| --- | --- | ---: | ---: | ---: | ---: |
| cstr | invariant_residual | 0.954061 | 0.982558 | 0.952800 | 0.000000 |
| cstr | repair_amount | 0.500000 | 0.720476 | 0.000000 | 1.000000 |
| two_tank | invariant_residual | 0.871416 | 0.937224 | 0.593421 | 0.000000 |
| two_tank | repair_amount | 0.500000 | 0.646667 | 0.000000 | 1.000000 |

## Accepted false accepts

| system | low_repair_bad | low_repair_high_invariant_bad | low_repair_low_invariant_bad |
| --- | ---: | ---: | ---: |
| cstr | 205 | 45 | 30 |
| two_tank | 108 | 49 | 15 |

## Interpretation

CSTR repair AUROC=0.500000; CSTR invariant AUROC=0.954061.

## Verdict

INVARIANT_DOMINATES_REPAIR
