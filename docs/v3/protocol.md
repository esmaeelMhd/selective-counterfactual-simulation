# v3 protocol

Status: draft, not frozen. Stage A may still set the numbers listed under "Open until the freeze". The freeze is a later commit that says the open table is closed and names its own hash. The commit that added this draft is not that freeze. Nothing here changes v1 or v2 artifacts or claims.

The reading of results, and any claim in a paper, is written by hand. This draft records the questions from `docs/v3/plan.md` and the working rules Stage A needs in order to compute a margin. Those working rules are marked below. They are not yet the confirmatory protocol.

## Question

When a learned simulator is asked a "what if" question outside its training range, can any refusal signal pick out the answers that are wrong, both for average error and for limit-violation events?

## Definitions

A scenario is one rollout question: an initial condition, a control and disturbance sequence, and the simulator's predicted trajectory.

Coverage is the share of judge-test scenarios accepted after sorting by risk score from lowest risk to highest risk.

False-accept rate (FAR) is the share of accepted scenarios whose prediction is bad under the active target.

Badness targets, same definitions as `docs/v2/benchmark_task.md`:

- `bad_rmse`: trajectory RMSE exceeds the threshold fixed for that system before the run.
- `bad_event`: predicted event status disagrees with true event status. In the current code that is one bit per trajectory, `true_event != pred_event`, not a count of limit crossings.

Primary coverages, carried from the v2 lock: 0.05 and 0.10. Both are reported. Whether a yes requires both, either, or a pooled margin is open until the freeze. Stage A does not collapse them.

## Decision questions

| ID | Question | Counts as yes when |
|---|---|---|
| Q1 | Do refusal signals reduce FAR for average error (`bad_rmse`)? | The calibration-selected judge beats `random_baseline` by the practical thresholds below, and the lower end of the bootstrap interval on that margin is above zero. |
| Q2 | Do the same signals reduce FAR for events (`bad_event`)? | Same rule, on the event target. |
| Q3 | Does the pre-registered event guard reduce event FAR? | It beats both the primary calibrated ranker and the best simple baseline chosen on calibration rows, each under the same rule, on `bad_event`. Q3 has no definition on BSM1 until the guard's missing-signal rule is written in this file. |

Every outcome is reported. "Q1 yes, Q2 no" is the v2 hypothesis confirmed. "Q3 no" is a negative result and still goes in the paper.

Practical thresholds from the plan, which keeps two of the three v2 cuts:

- absolute FAR reduction at least 0.05
- relative FAR reduction at least 0.10

The v2 lock also requires a seed win rate of at least 0.70. The plan's yes rule does not include it. Stage A reports the win rate only if a later edit adds it here. It is not part of the yes rule now.

The comparator for Q1 and Q2 is `random_baseline`, drawn without using test labels. The margin is random FAR minus judge FAR. This is not v2's `absolute_margin`. That column selects whichever simple or stronger baseline had the lowest FAR on the same test rows, including `conformal_risk_threshold` and `learned_error_classifier`.

The interval is a 95% bootstrap of the seed-level mean of that margin, resampling seeds with replacement, separately at each primary coverage. A yes requires the lower end above zero. An interval that sits entirely below zero excludes zero and is a no. If the random FAR is zero, the relative reduction is undefined and the relative half of the rule does not pass.

**Model eligibility, working definition.** A learned model counts on a system when its mean RMSE on `judge_calibration_id` is strictly below both `hold_last` and `linear_narx` on that system. Models that fail are reported and left out of the decision margins. Eligibility never reads a judge-test split: on BSM1 the test split does not exist until after the freeze, and a model must not be kept or dropped on test outcomes. This metric is a Stage A working rule and stays in the open table until the freeze.

## Judges

Stage A reports the simple baselines, `random_baseline`, `calibration_selected_candidate_ranker`, and the event guard. The ranker still chooses among the v2 candidate scores using calibration rows only. No new refusal method is added.

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

**Model eligibility.** See the working definition above.

**Sample size.** At least 2,000 judge-test scenarios per system, seed, and model. At least 8,000 where simulation is cheap. At 5% coverage, 2,000 scenarios make one accepted case move the false-accept rate by 0.01. The decision interval is over seeds. The seed list is what makes that interval narrow, and it is still open.

**Split separation.** Simulator models fit only on `model_train`. Judges and baselines fit only on judge-calibration rows. Effect estimates use only judge-test rows. If any calibration and test scenario hash overlaps, the run is invalid.

**Freeze before sample.** The freeze is a commit of this file and `configs/v3/stage_b.yaml` in which every open row has a value, and the commit message says the protocol is frozen. That commit happens before any confirmatory test data is generated, and its hash is written into the Stage B results. After that commit, the judge list, signal list, event-guard definition, coverage rule, thresholds, badness targets, and yes rules do not change. A later try is labelled exploratory. An earlier commit of this draft does not freeze it.

## Stages

**Stage A, exploratory.** TwoTank, CSTR, and heat exchanger. Used to fix model training, choose the event-guard definition, and set every tunable number. The Stage A report is labelled exploratory and does not issue the confirmatory yes or no.

**Stage B, confirmatory.** One new system, BSM1, after the freeze. Train on dry weather with controls in the nominal range. Test on rain, storm, control moves outside the training range, and the combination. Report once. No rerun with changed settings. The packaged dry file parses and is repeated. The packaged rain file does not parse, and the package has no storm file, so those two splits are not generated. Sampled windows do not overlap, and a four-day tail after the last sampled window stays unused until the confirmatory test split is allowed.

Signals that need the full state (`invariant_residual`, `repair_amount`) are marked not available on BSM1. A judge must still return a score when a signal is missing. The v2 event guard uses the invariant signal, so that guard is not the BSM1 guard. Q3 stays unanswered until this file names the guard that runs when invariant and repair are absent.

## What this protocol does not support

- safety certification
- a trusted simulator or a validated digital twin
- plant-wide validity
- high-coverage reliability
- product, service, or control readiness
- a method claim from Stage A

## Open until the freeze

A row with a Stage A value is a working setting. It is part of the freeze only when a later commit says the table is closed.

| Item | Now |
|---|---|
| Event-guard definition on the fully observed systems | v2 candidate above; Stage A may replace it |
| Event-guard definition on BSM1 | unset. Q3 cannot be answered until this row is filled |
| How 5% and 10% combine into one yes | unset. Stage A reports each coverage separately |
| Horizon, ensemble size, training stop | Stage A config sets a working value for the three ODE systems; not frozen |
| BSM1 horizon | working value, not frozen: 15 minutes of measured history and 480 minutes of prediction, one reactor hydraulic time |
| BSM1 event bit | working definition, not frozen: a trajectory is positive when any measured effluent series exceeds its benchmark limit (ammonia 4, total nitrogen 18, COD 100, suspended solids 30, BOD 10). This is not the BSM1 refusal guard |
| `bad_rmse` threshold per system | Stage A uses 0.05 for all three systems; not frozen |
| Eligibility metric | mean RMSE on `judge_calibration_id`, strict improvement on both baselines; never a judge-test split |
| Seed list | Stage A uses seeds 0–9; not frozen |
| BSM1 measured variables | working list, not frozen: ammonia, total nitrogen, COD, suspended solids, BOD |
| Bootstrap draw count | 500 |
