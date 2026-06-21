# CSTR Weakness Diagnosis

## Starting point

The current controlling claim is weak low-coverage support. Expansion is forbidden.

## Evidence summary

| analysis | verdict | key finding |
| --- | --- | --- |
| preconditions | READY_FOR_CSTR_WEAKNESS_AUDIT |  |
| diagnosis_table | ACCEPTED |  |
| statewise_error | BOTH_STATES | accepted false-accept normalized error share concentration=0.605476, temperature=0.394524 |
| repair_signal | REPAIR_SIGNAL_BLIND_SPOT | repair_amount AUROC=0.5, low-repair accepted false-accept fraction=1.000000 |
| signal_overlap | SIGNALS_SEPARATE_ACCEPTED_FAILURES | best deployable accepted-region signal is invariant_residual with AUROC 0.938531 |
| model_scenario_failure | DIFFUSE_CSTR_FAILURE | top model share=0.530303, top scenario share=0.227273 |
| rmse_target | CSTR_ACCEPTED_REGION_TOO_RISKY | 1 threshold/coverage cells pass practical thresholds; 53 available cells do not |

## Final diagnosis

REPAIR_SIGNAL_BLIND_SPOT

## Explanation

Diagnosis follows the fixed hierarchy in the CSTR weakness audit plan. The weak practical CSTR effect is not promoted to a broad claim.

## What not to do next

Do not add RSSM, a third system, heat exchanger evidence, a new judge, a new model, product/API/UI work, or a paper draft as a substitute for diagnosing CSTR weakness.

## Recommended next action

FIX_REPAIR_SIGNAL

## Allowed claims after this diagnosis

A weak but positive low-coverage result under the frozen protocol.

## Forbidden claims

strong two-system support, safety certification, product readiness, general reliable counterfactual simulation
