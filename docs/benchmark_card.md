# Benchmark Card

## Intended use

Use this benchmark to test whether a learned dynamical simulator can rank synthetic counterfactual intervention scenarios by answerability and reduce false accepts at fixed coverage.

## Non-intended use

Do not use this benchmark for safety certification, product validation, plant-wide deployment evidence, autonomous control, or claims of broad simulator reliability.

## Task

Given a simulator model and synthetic OOD/intervention scenarios, produce rollout predictions and rank scenarios from lowest risk to highest risk.

## Inputs

- synthetic train/calibration/test trajectory batches;
- states, actions, disturbances, and scenario labels;
- a built-in or custom simulator model implementing the adapter contract.

## Outputs

- risk-coverage tables;
- model and event metrics;
- accepted false-accept examples;
- benchmark summaries and plots.

## Metrics

The main metric is false accept rate at fixed coverage. Coverage is the accepted fraction after sorting scenarios by risk score.

## Systems

The public prototype includes synthetic TwoTank, CSTR, and heat-exchanger-oriented evidence paths. These are benchmark systems, not validated industrial simulators.

## Badness targets

- RMSE above threshold;
- event mismatch;
- RMSE-or-event failure.

## Current evidence

The current evidence supports a benchmark prototype, not a robust calibrated-refusal method claim. v1.1 is weak-positive at low coverage. v2 shows target-dependent behavior where event-risk remains a failure mode.

## Known limitations

The systems are synthetic, evidence is narrow, external validation is limited, and event-risk remains difficult. Large v2 CSV artifacts are retained only as frozen diagnostic evidence.

## Claim boundaries

This is not safety certification. This is not a product-ready digital twin. This is not a claim of general simulator reliability. This is not evidence that calibrated refusal works generally.
