# Task: Refusal Benchmark for Counterfactual Simulators

## Task summary

Given a simulator and counterfactual intervention scenarios, rank scenarios by answerability and accept only the lowest-risk fraction.

## Input

Synthetic train, calibration, and test trajectory batches with states, actions, disturbances, scenario labels, and intervention splits.

## Output

Per-scenario rollout predictions, error metrics, refusal/risk scores, and risk-coverage tables.

## Systems

Current evidence uses TwoTank and CSTR. TwoTank is stronger than CSTR.

## Scenario types

Normal policy, held-out action magnitude, step changes, inflow/feed spikes, degradation, combined interventions, and CSTR unsafe event scenarios.

## Models

Built-in baselines include hold-last, Linear NARX, and MLP state-space. User models can be added through the local adapter.

## Refusal signals

Support distance, uncertainty, disagreement, invariant residual, and repair amount. repair_amount is diagnostic-only for CSTR; invariant_residual is more informative there.

## Primary metric

False accept rate at fixed coverage.

## False accept definition

A false accept occurs when a judge accepts a scenario but the simulator rollout is bad under the configured error/event label.

## Coverage definition

Coverage is the fraction of scenarios accepted by the judge.

## Baselines

Support-only, uncertainty-only, disagreement-only, invariant-only, repair-only, random, oracle diagnostic, and calibrated/rank-based judges.

## Current best known result

Weak-positive at low coverage; TwoTank stronger than CSTR.

## How to submit/evaluate your own model locally

Implement `fit(train_batch)` and `predict_rollout(initial_state, actions, disturbances)`, then run `scripts/compare_models.py` with `--custom-model path.py:ClassName`.

## Fair comparison rules

Use the same splits, do not tune on test labels, report custom runs as local-only, and do not mix custom outputs into the frozen evidence claim.

## Non-goals

This is not production validation, safety certification, product readiness, autonomous control, plant-wide simulation, high-coverage reliability, or RSSM/third-system evidence.
