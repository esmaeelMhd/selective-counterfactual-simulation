# Benchmark Card

## Intended use

Use this benchmark to test whether a learned dynamical simulator can rank counterfactual intervention scenarios by answerability and reduce false accepts at fixed coverage.

## Non-intended use

Do not use this benchmark as safety certification, production validation, plant-wide deployment evidence, or proof of general simulator reliability.

## What this benchmark measures

It measures false accept rate at coverage for RMSE, event-risk, and RMSE-or-event targets under synthetic intervention shift.

## What it does not measure

It does not measure real plant safety, causal validity from observational data, cloud readiness, or autonomous control performance.

## Current v2 finding

Calibrated refusal is target-dependent and not reliable for event-risk under the current v2 protocol.

## Known weaknesses

Event-risk behavior is the main weakness. The row-wise strongest-baseline envelope is diagnostic only and not deployable.

## How to run

```bash
pip install -e ".[dev]"
pytest -q
python scripts/run_benchmark.py --model examples/custom_model_example.py:DampedLinearUserModel --output results/public_benchmark_run
```

## How to plug in a model

Implement `fit(train_batch)` and `predict_rollout(initial_state, actions, disturbances)` as described in `docs/v2/benchmark_api_contract.md`.

## Claim boundaries

The public benchmark exposes target-dependent calibrated-refusal failure. It does not support a robust calibrated-refusal method claim.
