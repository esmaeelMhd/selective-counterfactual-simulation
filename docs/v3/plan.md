# v3 plan: a powered, pre-registered refusal benchmark on a recognised process model

Status: WP0 drafted. WP1 Stage A has been run and is exploratory. WP2 has a checked BSM1 port, the measured-history API, effluent-limit events, and a dry-weather train and calibration dataset at an 8-hour horizon. WP3 adapters call the LSTM and the recurrent state-space model from time-series-simulator and train one step at a time. On the calibration rows their state RMSE is 122 and 176, and the Stage A delta MLP is 406, against 13.7 for `linear_narx`. A correction added to that linear step stays close for one minute and then the holdout rollout diverges, so that change was not kept. Eligibility is not decided, because `judge_test` is not generated. Rain and storm are not generated. The protocol is not frozen. Nothing in this plan changes v1 or v2 artifacts or claims.

## Goal

Answer one question with enough data to trust the answer, and publish it:

> When a learned simulator is asked a "what if" question outside its training range, can any
> refusal signal pick out the answers that are wrong, both for average error and for
> limit-violation events?

Done means three things exist:

1. A tagged v3.0 release that anyone can run on a laptop.
2. A short paper on arXiv, submitted to one venue.
3. At least three people outside this repo who have run their own simulator through it.

## Where the repo stands

- **v2 verdict.** `docs/v2/v2_scientific_summary.md` says `NO_METHOD_CLAIM_BENCHMARK_ONLY` and
  `NO_ROBUST_EFFECT`. The calibrated judge does not beat the best simple baseline chosen on
  calibration data (fair mean margin about -0.001).
- **v2 is too small to support any conclusion.** `configs/v2/v2_scientific_strengthening.yaml`
  uses 24 training trajectories, a horizon of 16 steps and ensembles of 2 members. Each
  system, seed and model has 50 to 70 test scenarios. At the primary coverages that is 3 or 4
  accepted scenarios at 5% and 5 to 7 at 10% (`v2_risk_coverage.csv`), so one wrong scenario
  moves the false-accept rate by 14 to 33 points. `reports/v2_statistical_audit.md` shows a
  bootstrap interval of zero width and the verdict `INVALID_DUE_TO_LEAKAGE_OR_BENCHMARK_FAILURE`.
- **The learned models are not yet real simulators.** In `reports/model_diversity_and_oracle_gap.md`
  the best model on TwoTank is `hold_last`, which predicts no change.
- **The event guard is untested.** `reports/v2_event_risk_fix_decision.md` allows a follow-up
  only if the guard is fixed as a candidate before new test outcomes are seen. v3 is that
  follow-up.
- **The systems are textbook.** TwoTank, CSTR and heat exchanger, all fully observed.
- **Process has outgrown evidence.** The repo holds about 100 scripts, 109 reports and 140
  test files for 23 commits of work.

So the "event risk fails" statement in the README is a hypothesis, not a finding. v3 tests it.

## Questions and decision rules

False-accept rate (FAR) is the share of accepted scenarios whose prediction is bad. Coverage
is the share of scenarios accepted.

| ID | Question | Counts as yes when |
|---|---|---|
| Q1 | Do refusal signals reduce FAR for average error (`bad_rmse`)? | The calibration-selected judge beats random by the v2 practical thresholds (absolute FAR reduction of at least 0.05, relative at least 10%), with a bootstrap interval that excludes zero. |
| Q2 | Do the same signals reduce FAR for events (`bad_event`)? | Same rule, on the event target. |
| Q3 | Does the pre-registered event guard reduce event FAR? | It beats both the primary calibrated ranker and the best calibration-selected simple baseline under the same rule. |

Every outcome is reported. "Q1 yes, Q2 no" is the v2 hypothesis confirmed. "Q3 no" is a
negative result and still goes in the paper.

Rules that apply to all three:

- **Model eligibility.** A learned model counts only if it beats both `hold_last` and
  `linear_narx` on in-distribution rollout error on that system.
- **Sample size.** At least 2,000 test scenarios per system, seed and model, and 8,000 where
  simulation is cheap. That gives 100 to 400 accepted scenarios at 5% coverage, against 3 or
  4 today.
- **Freeze before sample.** The protocol file is committed before any confirmatory test data
  is generated, and the commit hash is recorded in the results.

## Two stages

**Stage A, exploratory (existing systems).** Rerun the v2 design at proper scale on TwoTank,
CSTR and heat exchanger. Use it to fix model training, choose the event-guard definition and
set every tunable number. Results are labelled exploratory.

**Stage B, confirmatory (new system).** Freeze the protocol, then generate test data on a
system no judge has seen. Report once.

## Work packages

Hours assume 6 to 8 a week.

### WP0. Prior-work check and protocol draft (weeks 1–2, about 12 h)

- Search for existing work on selective prediction and risk-coverage curves, conformal
  prediction for dynamical models, and uncertainty in learned dynamics for offline
  model-based reinforcement learning.
- Write a half-page statement of what this benchmark measures that those do not.
- Draft `docs/v3/protocol.md` with the questions and rules above.
- Done when: the half page exists and names the closest three prior works.

### WP1. Stage A: fix the scale (weeks 2–4, about 15 h)

- Add one config, `configs/v3/stage_a.yaml`, with the larger sample sizes, longer horizon,
  ensembles of at least 5 members and models trained to convergence.
- Add one runner, `scripts/v3_run.py`, that takes a config and writes one results table and
  one report.
- Run on the three existing systems. These are small ODEs, so this is minutes on a laptop.
- Done when: the model eligibility rule passes on at least two systems, and FAR at 5%
  coverage has an interval narrower than 0.10.

### WP2. Interface change and the new system (weeks 5–8, about 30 h)

- **New system: BSM1**, the standard activated-sludge wastewater benchmark. It defines its
  own events (effluent limits on ammonia, total nitrogen, COD, suspended solids and BOD) and
  its own shifted conditions (dry, rain and storm influent files).
- Splits: train on dry weather with controls in the nominal range. Test on rain, storm,
  control moves outside the training range, and the combination.
- Use an existing open Python implementation. Check its licence, check it against the
  published BSM1 steady-state values, and time one run before generating data.
- **Interface change.** BSM1 has about 145 internal states and only some are measured. The
  current API passes one full initial state. Add an optional history window of measured
  variables to `TrajectoryBatch` and `predict_rollout`, keeping the old call working.
- Signals that need the full state (`invariant_residual`, `repair_amount`) are marked not
  available on BSM1. Judges must handle a missing signal.
- Publish the generated dataset as a release asset. It is synthetic, so it can be public.
- Done when: `pytest` passes, old v2 runs still reproduce, and a BSM1 dataset with all
  splits exists except the confirmatory test split.

### WP3. Learned simulators (weeks 8–11, about 25 h)

- Add adapters for the LSTM and the recurrent world model from `time-series-simulator`, plus
  a deep ensemble of the LSTM for uncertainty.
- Keep PyTorch as an optional extra so the core install stays light.
- Stretch only: the neural SDE from `digital-twin-engine`. It adds a second framework.
- Done when: at least two learned models pass the eligibility rule on BSM1.

### WP4. Stage B: confirmatory run (weeks 12–13, about 12 h)

- Freeze `docs/v3/protocol.md` and `configs/v3/stage_b.yaml` and commit.
- Generate the confirmatory test split, run once, write the report.
- No reruns with changed settings. Anything tried afterwards is labelled exploratory.

### WP5. Paper, release, users (weeks 14–17, about 30 h)

- Write a short paper: the question, the protocol, Stage A, Stage B, limits.
- Tag v3.0. Replace the README evidence section with the v3 result.
- Post to arXiv and submit to one workshop or journal, chosen in week 12.
- Ask three to five people who build process simulators to run their model and send the
  results table.

## Optional: real-plant section

Real plant data has no "what if" ground truth, so it cannot be a main result. One section
could test refusal on unusual historical periods of one wastewater plant dataset that stays outside this repository.

- Only with written permission from the data owner to publish aggregate numbers.
- No rows, plots of raw series or derived datasets enter this repo.
- If permission is not in hand by week 8, the section is dropped.

## Out of scope

- New model architectures or new refusal methods beyond the one event guard.
- More than one new system.
- Any product, service or control claim.
- New gate, audit or packaging scripts. v3 has one protocol, one config per stage, one runner
  and one report per stage.

## Budget

| Item | Estimate |
|---|---|
| Time | About 125 hours over 17 weeks |
| Compute | Laptop. A rented GPU is needed only if recurrent models train too slowly; estimate under 30 euros in total |
| Hosting | Free (GitHub release assets, arXiv) |

## Risks and stop rules

| Risk | Rule |
|---|---|
| Someone has already built this | If WP0 finds a benchmark with the same measurement, reposition v3 as an extension of it or stop. |
| No learned model beats `hold_last` | One week to fix scenario design in Stage A. If still failing, publish the existing technical note as it is and stop. |
| BSM1 does not run or validate | If not working after three weeks of WP2, switch to another standard process benchmark with a maintained Python port. |
| Scope creep | No new repo, system or method until the paper is submitted. |
| Agent-written claims | Coding agents write adapters and tests. The protocol, the results reading and the paper claims are written by hand. |

## Open decisions

| Decision | Default |
|---|---|
| Real-plant section | Out, unless permission arrives by week 8 |
| Neural SDE model | Stretch goal, only after WP3 is done |
| Venue | arXiv plus one workshop or journal, chosen in week 12 |
