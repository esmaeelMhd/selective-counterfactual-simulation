# v3 protocol

Status: draft, not frozen. Stage A may still set the numbers listed under "Open until the freeze". This file becomes the confirmatory protocol only when it is committed before any Stage B test data is generated. That commit hash is recorded in the Stage B results. Nothing here changes v1 or v2 artifacts or claims.

The reading of results, and any claim in a paper, is written by hand. This draft only records the questions and rules already stated in `docs/v3/plan.md`.

## Question

When a learned simulator is asked a "what if" question outside its training range, can any refusal signal pick out the answers that are wrong, both for average error and for limit-violation events?

## Definitions

A scenario is one rollout question: an initial condition, a control and disturbance sequence, and the simulator's predicted trajectory.

Coverage is the share of judge-test scenarios accepted after sorting by risk score from lowest risk to highest risk.

False-accept rate (FAR) is the share of accepted scenarios whose prediction is bad under the active target.

Badness targets, same definitions as `docs/v2/benchmark_task.md`:

- `bad_rmse`: trajectory RMSE exceeds the threshold fixed for that system before the run.
- `bad_event`: predicted event status disagrees with true event status.

Primary coverages, carried from the v2 lock: 0.05 and 0.10. Both are reported. A question is decided on these two, not on the rest of the curve.

## Decision questions

| ID | Question | Counts as yes when |
|---|---|---|
| Q1 | Do refusal signals reduce FAR for average error (`bad_rmse`)? | The calibration-selected judge beats random by the practical thresholds below, and the bootstrap interval on the FAR margin excludes zero on the positive side. |
| Q2 | Do the same signals reduce FAR for events (`bad_event`)? | Same rule, on the event target. |
| Q3 | Does the pre-registered event guard reduce event FAR? | It beats both the primary calibrated ranker and the best calibration-selected simple baseline, each under the same rule, on `bad_event`. |

Every outcome is reported. "Q1 yes, Q2 no" is the v2 hypothesis confirmed. "Q3 no" is a negative result and still goes in the paper.

Practical thresholds, carried from `docs/v2/v2_scientific_protocol_lock.md`:

- absolute FAR reduction at least 0.05
- relative FAR reduction at least 0.10

The interval is a 95% bootstrap interval of the seed-level mean absolute FAR margin, resampling seeds with replacement, as in the v2 statistical audit. "Excludes zero" means the lower end is above zero. The v2 seed-win-rate cutoff is reported and is not part of the yes rule.

## Judges

Signals and judges are the v2 lists. No new refusal method is added.

Simple baselines, one per signal, plus `random_baseline`:

- `support_only`
- `uncertainty_only`
- `disagreement_only`
- `invariant_only`
- `repair_only`

Primary calibrated judge: `calibration_selected_candidate_ranker`. It is fit only on judge-calibration rows.

Event-guard candidate, starting definition, taken from `configs/v2/v2_event_risk_fix.yaml` and `event_guarded_score` in `src/scs/experiments/v2_event_risk.py`: the score is the maximum calibration-percentile rank of `risk_invariant_only`, `risk_disagreement_only`, and `risk_support_only`. Lower is safer. A scenario is low risk only when all three are low relative to calibration. Stage A may replace this definition. The replacement is written into this file before the freeze. No new raw signal is allowed in the replacement.

`oracle_error_rank` is diagnostic only and cannot count as a judge.

## Rules that apply to all three questions

**Model eligibility.** A learned model counts only if it beats both `hold_last` and `linear_narx` on in-distribution rollout error on that system. Models that fail are reported and dropped from the decision.

**Sample size.** At least 2,000 judge-test scenarios per system, seed, and eligible model. At least 8,000 where simulation is cheap. The point is 100 to 400 accepted scenarios at 5% coverage, against 3 or 4 in v2.

**Split separation.** Simulator models fit only on `model_train`. Judges and baselines fit only on judge-calibration rows. Effect estimates use only judge-test rows. If any calibration and test scenario hash overlaps, the run is invalid.

**Freeze before sample.** This file and `configs/v3/stage_b.yaml` are committed before any confirmatory test data is generated. The commit hash is written into the Stage B results. After that commit, the judge list, signal list, event-guard definition, coverage grid, thresholds, badness targets, and yes rules do not change. A later try is labelled exploratory.

## Stages

**Stage A, exploratory.** TwoTank, CSTR, and heat exchanger. Used to fix model training, choose the event-guard definition, and set every tunable number. Results are labelled exploratory and cannot answer Q1–Q3.

**Stage B, confirmatory.** One new system, BSM1, after the freeze. Train on dry weather with controls in the nominal range. Test on rain, storm, control moves outside the training range, and the combination. Report once. No rerun with changed settings.

Signals that need the full state (`invariant_residual`, `repair_amount`) are marked not available on BSM1. A judge must still return a score when a signal is missing. Those two signals are then absent from the BSM1 baselines, and the event guard on BSM1 cannot use them.

## What this protocol does not support

- safety certification
- a trusted simulator or a validated digital twin
- plant-wide validity
- high-coverage reliability
- product, service, or control readiness
- a method claim from Stage A

## Open until the freeze

Stage A writes the chosen value into this file. Until then the line stays unset.

| Item | Now |
|---|---|
| Event-guard definition | v2 candidate above; Stage A may replace it |
| Horizon, ensemble size, training stop | unset |
| `bad_rmse` threshold per system | unset |
| Seed list | unset |
| BSM1 measured variables | unset |
| Bootstrap seed count | 500, matching v2, unless Stage A changes it here |
