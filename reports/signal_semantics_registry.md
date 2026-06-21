# Signal Semantics Registry

This registry describes what each refusal signal can and cannot detect. No signal is treated as universally reliable.

## support_distance

Description: Distance between an action/disturbance trajectory and the model-training support.

Risk orientation: higher_is_riskier

Expected higher means: trajectory is farther from the empirical training action/disturbance distribution

| field | value |
| --- | --- |
| requires_trajectory | True |
| requires_bounds | False |
| requires_repair_operator | False |
| requires_ensemble | False |
| cstr_role | candidate_context_signal |
| twotank_role | candidate_context_signal |
| universal_refusal_signal | False |
| is_universal_candidate | False |
| is_system_specific_candidate | True |

CSTR note: Useful for CSTR support shift, but not a universal refusal signal.

System applicability:
- two_tank: applicable when action/disturbance support shift is a failure driver
- cstr: applicable when feed/cooling disturbance support shift is a failure driver

Failure types detected:
- extrapolation outside observed action/disturbance support
- intervention magnitude shift

Failure types not detected:
- within-support dynamics model error
- wrong state trajectory when inputs remain in support
- state-bound violation by itself

Known blind spots:
- does not inspect predicted state consistency
- can be low for hard in-distribution model errors

## uncertainty_score

Description: Mean predictive spread from rollout samples for a single simulator model.

Risk orientation: higher_is_riskier

Expected higher means: model sample rollouts disagree with each other

| field | value |
| --- | --- |
| requires_trajectory | True |
| requires_bounds | False |
| requires_repair_operator | False |
| requires_ensemble | True |
| cstr_role | candidate_model_uncertainty_signal |
| twotank_role | candidate_model_uncertainty_signal |
| universal_refusal_signal | False |
| is_universal_candidate | False |
| is_system_specific_candidate | True |

CSTR note: Not selected as the strongest current CSTR accepted-region separator.

System applicability:
- two_tank: applicable if model sampling captures epistemic or rollout uncertainty
- cstr: applicable if model sampling captures reaction/temperature uncertainty

Failure types detected:
- model self-uncertainty
- unstable sampled rollouts

Failure types not detected:
- confident but biased dynamics
- deterministic model misspecification
- state-bound violation when all samples agree

Known blind spots:
- deterministic or under-dispersed models can produce low uncertainty on wrong predictions
- sample noise is not a calibrated probability of RMSE failure

## disagreement_score

Description: Mean trajectory disagreement across model predictions for the same scenario.

Risk orientation: higher_is_riskier

Expected higher means: available simulator models predict meaningfully different trajectories

| field | value |
| --- | --- |
| requires_trajectory | True |
| requires_bounds | False |
| requires_repair_operator | False |
| requires_ensemble | True |
| cstr_role | candidate_model_disagreement_signal |
| twotank_role | candidate_model_disagreement_signal |
| universal_refusal_signal | False |
| is_universal_candidate | False |
| is_system_specific_candidate | True |

CSTR note: Mixed CSTR separability in the CSTR weakness audit.

System applicability:
- two_tank: applicable when model class diversity exposes intervention-shift failure
- cstr: applicable when model class diversity exposes reaction or thermal dynamics failure

Failure types detected:
- model-class disagreement
- ambiguous rollout behavior under the same intervention

Failure types not detected:
- shared bias across models
- wrong predictions when all models agree

Known blind spots:
- ensembles with common bias can disagree little while all being wrong
- large disagreement is not specific to the physical cause of failure

## invariant_residual

Description: Residual from system-specific physical accounting or dynamics consistency checks.

Risk orientation: higher_is_riskier

Expected higher means: predicted trajectory is inconsistent with known system dynamics or accounting

| field | value |
| --- | --- |
| requires_trajectory | True |
| requires_bounds | False |
| requires_repair_operator | False |
| requires_ensemble | False |
| cstr_role | informative_refusal_signal |
| twotank_role | informative_refusal_signal |
| universal_refusal_signal | False |
| is_universal_candidate | False |
| is_system_specific_candidate | True |

CSTR note: Best accepted-region signal in CSTR weakness audit.

System applicability:
- two_tank: inventory accounting residual for external inflow/outflow consistency
- cstr: one-step CSTR concentration/temperature dynamics residual under known equations

Failure types detected:
- dynamics/accounting inconsistency
- trajectory that cannot be reproduced by the known system step

Failure types not detected:
- all possible trajectory errors
- errors that satisfy the checked invariant
- input support shift by itself

Known blind spots:
- only detects violations of the implemented invariant
- does not prove the trajectory is causally correct

## repair_amount

Description: Magnitude of clipping/projection needed to bring a predicted state trajectory back inside system bounds.

Risk orientation: higher_is_riskier

Expected higher means: predicted states violate configured physical bounds and require repair

| field | value |
| --- | --- |
| requires_trajectory | True |
| requires_bounds | True |
| requires_repair_operator | True |
| requires_ensemble | False |
| cstr_role | diagnostic_only |
| twotank_role | diagnostic_constraint_signal |
| universal_refusal_signal | False |
| is_universal_candidate | False |
| is_system_specific_candidate | True |

CSTR note: Correct as a bounds/projection signal but irrelevant for within-bound CSTR dynamic errors.

System applicability:
- two_tank: applicable to negative inventory and over-capacity predictions
- cstr: applicable only to concentration/temperature predictions outside configured bounds

Failure types detected:
- state-bound violation
- projection or clipping correction amount
- negative inventory or over-capacity state for bounded systems

Failure types not detected:
- within-bound dynamical error
- wrong reaction rate
- wrong temperature trajectory inside bounds
- wrong concentration trajectory inside bounds
- support shift without state-bound violation

Known blind spots:
- within-bound dynamic errors can have high RMSE and zero repair
- bounded but physically inconsistent trajectories are not detected unless they leave bounds
- constant zero repair provides no ranking information for refusal
