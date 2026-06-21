# Effect Size and Uncertainty Audit

## Question

Is the calibrated low-coverage improvement statistically and practically meaningful?

## Fixed practical thresholds

minimum_absolute_far_reduction: 0.05
minimum_relative_far_reduction: 0.1
confidence_level: 0.95
bootstrap_iterations: 1000

## Main result

| system | coverage | absolute_margin | relative_margin | bootstrap_95_ci | seed_95_ci | seed_win_rate | practical_threshold_passed | verdict |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: | --- |
| two_tank | 0.050000 | 0.173333 | 0.270833 | [0.146667, 0.200000] | [0.157465, 0.186535] | 1.000000 | True | PRACTICALLY_MEANINGFUL |
| two_tank | 0.100000 | 0.166667 | 0.255102 | [0.113333, 0.193333] | [0.121105, 0.144229] | 1.000000 | True | PRACTICALLY_MEANINGFUL |
| cstr | 0.050000 | 0.038095 | 0.057143 | [0.019048, 0.047619] | [0.004588, 0.027792] | 0.700000 | False | POSITIVE_BUT_WEAK |
| cstr | 0.100000 | 0.038095 | 0.057143 | [0.019048, 0.042857] | [0.001231, 0.025436] | 0.600000 | False | POSITIVE_BUT_WEAK |

## TwoTank interpretation

- coverage 0.05: margin 0.173333, relative 0.270833, seed win rate 1.000000, verdict PRACTICALLY_MEANINGFUL
- coverage 0.1: margin 0.166667, relative 0.255102, seed win rate 1.000000, verdict PRACTICALLY_MEANINGFUL

## CSTR interpretation

- coverage 0.05: margin 0.038095, relative 0.057143, seed win rate 0.700000, verdict POSITIVE_BUT_WEAK
- coverage 0.1: margin 0.038095, relative 0.057143, seed win rate 0.600000, verdict POSITIVE_BUT_WEAK

## Cross-system interpretation

The audit keeps the TwoTank and CSTR margins separate. A small CSTR margin is not promoted to a strong practical effect.

## Verdict

WEAK_TWO_SYSTEM_EFFECT

## Explanation

The final verdict follows fixed practical thresholds from `configs/audits/effect_size_audit.yaml`.

## Known limitations

Bootstrap samples scenario-model rows within each model/scenario group.; Seed-level calibrated margin uses the frozen calibrated family outputs from each seed sweep.; The audit quantifies existing evidence and does not tune the judge.

## Reproduction command

```bash
python scripts/analyze_effect_size_uncertainty.py --config configs/audits/effect_size_audit.yaml --output results/effect_size_audit/effect_size
```
