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
- `bad_event`: predicted event status disagrees with true event status. On the three ODE systems that is one bit per trajectory, `true_event != pred_event`, not a count of limit crossings. On BSM1 each of the five effluent limits is its own bit, and a scenario is bad when any of the five bits differs. The combined any-limit bit is not used there, because on the dry calibration rows it never disagrees: the open-loop plant is nearly always over the ammonia limit.

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

**Stage B, confirmatory.** One new system, BSM1, after the freeze. Train on dry weather with controls in the nominal range. Test on rain, control moves outside the training range, and the combination. Storm is dropped. Report once. No rerun with changed settings. The packaged dry file lists 14 days, but days 7 to 13 are an exact copy of days 0 to 6, so it holds 7 unique days. The code checks this and reads only those 7. Each pretest split reads its own block of them: days 0–3.5 for `model_train`, 3.5–4.5 for `judge_calibration_id`, and 4.5–5.5 for `judge_calibration_control`. No two splits share an influent minute. Windows inside one split may overlap, so the number of independent influent windows is far below the window count. The manifest lists it per split. Days 5.5–7 are not read until the confirmatory test split is allowed. Every window starts from the plant state at its own minute of one continuous dry run at nominal aeration, after a 14-day warm-up.

Each window also gets one influent flow factor and one load factor, drawn uniformly within ±10% and held constant over the window. Concentrations and total suspended solids scale with the load factor. Alkalinity and temperature do not. The ±10% matches the day-to-day spread of daily mean flow in the dry file. The same scheme applies to every split, including the confirmatory test, and the manifest records each window's factors. It adds variety in level, not new daily shapes, so the independent-window counts in the manifest still describe the shapes.

The packaged rain file has three malformed flow cells, and its flow is in thousands of m³/d. The loader repairs both. Each bad cell has a stray second decimal point, and without it the value lies on the straight line between its readable neighbours, which the code checks. The only rain event is on days 8–10 of that file, and those days belong to the confirmatory test. No rain split is generated before the freeze, so judge calibration uses the control shift only. The package has no storm file, and the public benchmark repository checked has none, so storm is dropped from the confirmatory test. It comes back only if an official file is supplied before the freeze.

Signals that need the full state (`invariant_residual`, `repair_amount`) are marked not available on BSM1. A judge must still return a score when a signal is missing. The v2 event guard uses the invariant signal, so that guard is not the BSM1 guard. The runner does not compute the guard on the inputs that remain, because that would be a different guard. It reports the guard as undefined on that system. Q3 stays unanswered until this file names the guard that runs when invariant and repair are absent.

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
| BSM1 dry-file blocks | working value, not frozen, of the 7 unique days: train 0–3.5, calibration ID 3.5–4.5, calibration control 4.5–5.5, reserved 5.5–7 |
| BSM1 weather in the confirmatory test | rain: repaired file, days 8–10 reserved. Storm: dropped, no official file in the package or the public benchmark repository checked |
| BSM1 influent perturbation | working value, not frozen: flow and load factors each uniform in 0.9 to 1.1 per window, every split |
| BSM1 event bit | working definition, not frozen: each of the five effluent limits is a bit (ammonia 4, total nitrogen 18, COD 100, suspended solids 30, BOD 10), true when any measured value in the trajectory exceeds it, and a scenario is bad for `bad_event` when any of the five bits differs between truth and prediction. This is not the BSM1 refusal guard. On the dry pretest windows ammonia is over its limit in 93% to 100% of windows and total nitrogen in 25% to 43%. The other three limits are never exceeded. The combined any-limit bit never disagrees on the calibration rows. The per-limit target does, from total nitrogen only: on one seed, 0% to 1% for `linear_narx`, up to 41% for the delta MLP, and 11% to 18% for the LSTM. The limits are the standard ones and are not changed. Nominal aeration stays at the open-loop setting |
| `bad_rmse` threshold per system | Stage A uses TwoTank 0.08, CSTR 0.38, heat exchanger 0.095, not frozen. Each is 10 times the 99th percentile of the best model's RMSE on `judge_calibration_id`, over seeds 0–2, from calibration rows only |
| BSM1 `bad_rmse` threshold | working value, not frozen: 61.761. It is 3 times the 99th percentile (20.5869) of `ensemble_lstm` per-scenario RMSE on `judge_calibration_id`, seeds 0–2. The multiple 3 is the smallest whole number that no eligible model's in-distribution calibration scenario exceeds (largest 41.6, the LSTM). The multiple of 10 used on the ODE systems gives 205.869, which marks no scenario bad for any eligible model on either calibration split. Calibration rows only. On the calibration rows the threshold marks 7% of the scenarios of the eligible models as bad, so the share on the confirmatory test depends on how far the test shifts go |
| Degenerate target | a 5% row is reported as degenerate, and does not count toward the precision check, when under 1% or over 99% of its test scenarios are bad |
| Eligibility metric | mean RMSE on `judge_calibration_id`, strict improvement on both baselines; never a judge-test split |
| BSM1 learned-model budgets | working rule, not frozen: LSTM, LSTM ensemble (5 members) and recurrent model use early stopping on the last 15% of the `model_train` windows, less a gap of 18 windows (the number of neighbours that overlap one window in time). The free-running rollout error with the BSM1 flow assignment is checked every 5 epochs, the best weights are kept, and training stops after 3 checks without improvement or at 120 epochs. Neither calibration split is read. The delta MLP keeps 80 iterations from before this work. Seeds 0–2 |
| Seed list | Stage A uses seeds 0–9; not frozen |
| BSM1 measured variables | working list, not frozen: ammonia, total nitrogen, COD, suspended solids, BOD |
| Bootstrap draw count | 500 |
