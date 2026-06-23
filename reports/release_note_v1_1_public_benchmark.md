# v1.1 Public Benchmark Release

## What this is

First public release of the Selective Counterfactual Simulation Benchmark.

## Main purpose

A benchmark for testing whether learned dynamical simulators know when to refuse counterfactual predictions.

## Current evidence

Narrow synthetic evidence only. The repo is best treated as a benchmark prototype.

## What is not claimed

This is not safety certification, not a product-ready digital twin, and not a claim of general simulator reliability.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
python scripts/run_smoke.py
python scripts/run_current_status_demo.py
python examples/custom_model_example.py --output results/custom_model_example
```

## Known limitations

Synthetic systems, limited external validation, narrow low-coverage evidence, and event-risk remains a hard failure mode under v2.
