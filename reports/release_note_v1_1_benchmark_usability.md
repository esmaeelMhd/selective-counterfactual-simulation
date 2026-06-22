# Release Note: v1.1 Benchmark Usability

## What changed

Added a quickstart demo, benchmark card, custom model adapter, custom model example, local model comparison, README usability sections, and a usability package checker.

## What did not change

The scientific claim did not change. Expansion remains blocked.

## Current allowed claim

A weak but positive low-coverage refusal benchmark under a frozen protocol.

## New quickstart demo

Run `python scripts/run_current_status_demo.py --config configs/status/benchmark_usability_v1_1.yaml --output results/demo`.

## Custom model adapter

Use `src/scs/models/user_model.py` and `examples/custom_model_example.py`.

## Local model comparison

Use `scripts/compare_models.py` for built-in or custom models. Results are local-only.

## Claim boundaries

Do not claim strong support, safety certification, product readiness, high-coverage reliability, RSSM evidence, or third-system evidence.

## Reproducibility

Run `pip install -e ".[dev]"`, `pytest -q`, the demo command, and the model comparison command from the README.

## What not to do next

Do not treat usability features as new benchmark evidence. Do not add RSSM, third-system, product/API/frontend, or deployment claims.
