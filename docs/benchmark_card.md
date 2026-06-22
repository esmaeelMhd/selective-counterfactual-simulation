# Benchmark Card: Selective Counterfactual Simulation

## Intended use

Use this benchmark to test refusal/ranking behavior for counterfactual simulator rollouts under intervention shift.

## Non-intended use

This benchmark tests refusal/ranking behavior, not simulator safety. It is not intended for product use, certification, autonomous control, or plant-wide deployment claims.

## Core question

Can a simulator identify which counterfactual intervention scenarios it can answer reliably and abstain on the rest?

## Systems included

The current evidence package includes TwoTank and CSTR. TwoTank is stronger than CSTR. Expansion to RSSM, third systems, or product use is not currently supported.

## Models included

Built-in local models are `hold_last`, `linear_narx`, and `mlp_state_space`.

## Refusal signals included

Support distance, uncertainty, disagreement, invariant residual, and repair amount are available. repair_amount is diagnostic-only for CSTR. invariant_residual is informative for CSTR.

## Primary metric

False accept rate at fixed coverage.

## What counts as a false accept

A false accept occurs when a judge accepts a scenario whose simulator prediction is materially wrong under the configured error threshold.

## Current evidence status

A weak but positive low-coverage refusal benchmark under a frozen protocol. Current evidence is weak-positive and low-coverage only. TwoTank margin at coverage 0.05 is 0.173333; CSTR margin at coverage 0.05 is 0.038095.

## Known weaknesses

CSTR is positive but weak. repair_amount misses within-bound CSTR dynamic errors, while invariant_residual is more informative.

## How to run the quickstart demo

```bash
python scripts/run_current_status_demo.py --config configs/status/benchmark_usability_v1_1.yaml --output results/demo
```

## How to add a custom model

Implement `fit(train_batch)` and `predict_rollout(initial_state, actions, disturbances)` using `src/scs/models/user_model.py`, then run `python examples/custom_model_example.py --output results/custom_model_example`.

## How to compare models fairly

Use the local comparison script and report results as local-only:

```bash
python scripts/compare_models.py --config configs/experiments/calibrated_two_tank.yaml --models hold_last linear_narx mlp_state_space --output results/model_comparison
```

## Claim boundaries

Do not claim strong support, broad simulator reliability, safety certification, product readiness, high-coverage reliability, RSSM evidence, or third-system evidence.

## Reproducibility

Run `pip install -e ".[dev]"`, `pytest -q`, and the quickstart demo command above.
