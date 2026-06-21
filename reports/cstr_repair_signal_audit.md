# CSTR Repair-Signal Blind-Spot Audit

## Question

Does repair_amount protect against CSTR false accepts?

## Repair distribution

| group | mean | median | p10 | p90 | zero_fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| accepted_good | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 |
| accepted_bad | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 |
| rejected_good | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 |
| rejected_bad | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 |

## Repair as failure detector

| metric | value |
| --- | ---: |
| repair_auroc_for_bad_rmse_label | 0.500000 |
| repair_auprc_for_bad_rmse_label | 0.766462 |
| repair_spearman_with_rmse | 0.000000 |
| repair_false_negative_rate_at_low_risk_threshold | 1.000000 |
| fraction_accepted_false_accepts_with_low_repair | 1.000000 |
| fraction_severe_false_accepts_with_low_repair | 1.000000 |
| zero_repair_fraction | 1.000000 |
| near_zero_repair_fraction | 1.000000 |
| repair_dynamic_range | 0.000000 |

## Low-repair false accepts

| coverage | accepted_false_accept_count | low_repair_false_accept_count | fraction |
| ---: | ---: | ---: | ---: |
| 0.050000 | 66.000000 | 66.000000 | 1.000000 |
| 0.100000 | 132.000000 | 132.000000 | 1.000000 |

## Repair-only judge comparison

| coverage | repair_only_far | calibrated_far |
| ---: | ---: | ---: |
| 0.050000 | 0.704762 | 0.628571 |
| 0.100000 | 0.728571 | 0.628571 |

## Interpretation

repair_amount AUROC=0.5, low-repair accepted false-accept fraction=1.000000

## Verdict

REPAIR_SIGNAL_BLIND_SPOT
