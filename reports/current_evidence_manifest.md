# Current Evidence Manifest

## Controlling status

NARROW_TO_WEAK_LOW_COVERAGE_CLAIM

## Allowed claim

A weak but positive low-coverage result under the frozen protocol.

## Forbidden claims

strong two-system support, broad simulator reliability, trustworthy counterfactual simulation, safety certification, autonomous control, product readiness, plant-wide digital twin, industrial AI breakthrough, high-coverage reliability

## Expansion status

Expansion allowed: False

## TwoTank evidence

| system | coverage_0_05_margin | coverage_0_10_margin | effect_strength |
| --- | ---: | ---: | --- |
| two_tank | 0.173333 | 0.166667 | practically_meaningful |

## CSTR evidence

| system | coverage_0_05_margin | coverage_0_10_margin | effect_strength |
| --- | ---: | ---: | --- |
| cstr | 0.038095 | 0.038095 | positive_but_weak |

## Repair signal role

diagnostic_only: Correct as a bounds/projection signal but irrelevant for within-bound CSTR dynamic errors.

## Invariant residual role

informative_refusal_signal: Dominates repair_amount on CSTR accepted-region separation.

## Allowed next action

UPDATE_SIGNAL_SEMANTICS_ONLY

## Source artifacts

| artifact | path |
| --- | --- |
| twotank_calibrated_summary | results/calibrated_two_tank/calibrated_judge_summary.json |
| twotank_low_coverage | results/calibrated_two_tank/low_coverage_summary.csv |
| cstr_calibrated_summary | results/calibrated_cstr/calibrated_judge_summary.json |
| cstr_low_coverage | results/calibrated_cstr/low_coverage_summary.csv |
| effect_size_summary | results/effect_size_audit/effect_size/effect_size_summary.json |
| cstr_weakness_summary | results/cstr_weakness_audit/repair_signal/repair_signal_metrics.json |
| repair_role_summary | results/repair_signal_semantics_audit/repair_validation/repair_validation_summary.json |
| repair_vs_invariant_summary | results/repair_signal_semantics_audit/repair_vs_invariant/repair_vs_invariant_summary.json |
| repair_signal_decision | reports/repair_signal_role_decision_gate.md |
