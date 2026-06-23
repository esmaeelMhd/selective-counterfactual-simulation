# Selective Counterfactual Simulation Benchmark

A benchmark for testing whether learned dynamical simulators know when to refuse counterfactual predictions.

Plug in a simulator, run OOD/intervention scenarios, and compare false-accept rate versus coverage.

**Current status:** this is a benchmark prototype with narrow synthetic evidence. It is not a safety tool, product-ready digital twin, or claim of general simulator reliability.

## What this is

Selective Counterfactual Simulation is a small Python benchmark for refusal/ranking behavior in learned dynamical simulators. It generates synthetic intervention-shift trajectories, trains or loads simulator models, ranks scenarios by risk, and reports false-accept rate at fixed coverage.

The central question is:

> Can a simulator identify which counterfactual intervention scenarios it can answer reliably and abstain on the rest?

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

## Run the benchmark/demo

The fastest public demo is:

```bash
python scripts/run_current_status_demo.py
```

The original TwoTank smoke benchmark is:

```bash
python scripts/run_smoke.py
```

The public plug-in benchmark command is:

```bash
python scripts/run_benchmark.py --model examples/custom_model_example.py:DampedLinearUserModel --output results/public_benchmark_run
```

## Plug in your own model

Implement a class with:

```python
model_id: str

def fit(self, train_batch) -> None:
    ...

def predict_rollout(self, initial_state, actions, disturbances):
    ...
```

Then run:

```bash
python scripts/run_benchmark.py --model path/to/file.py:ClassName --output results/my_benchmark_run
```

See `examples/custom_model_example.py` and `docs/custom_model_adapter.md`.

## Current evidence

The current evidence supports a benchmark prototype, not a robust calibrated-refusal method claim.

v1.1 shows weak-positive low-coverage behavior under a frozen synthetic protocol.

v2 harder evidence shows target-dependent behavior: RMSE can look near-neutral while event-risk remains a failure mode. In short, calibrated refusal is target-dependent and not reliable for event-risk.

Public benchmark outputs include:

- `results/public_benchmark_run/risk_coverage.csv`
- `results/public_benchmark_run/risk_coverage.png`
- `results/public_benchmark_run/benchmark_summary.json`
- `docs/v2/figures/event_risk_vs_rmse_public.png`
- `docs/v2/event_risk_failure_gallery.md`

## What is not claimed

This is not safety certification.

This is not a product-ready digital twin.

This is not a claim of general simulator reliability.

This is not an autonomous control system.

This is not evidence that calibrated refusal works generally.

This repository should not be described as a trusted simulator, a validated industrial simulator, or a production-ready industrial AI system.

## Repository structure

- `src/scs/systems/`: synthetic dynamical systems.
- `src/scs/models/`: simulator model interfaces and lightweight baselines.
- `src/scs/validators/`: refusal/risk signals.
- `src/scs/metrics/`: trajectory, event, and selective risk metrics.
- `src/scs/experiments/`: reproducible experiment and release-check logic.
- `configs/`: experiment and audit configs.
- `examples/`: custom model adapter examples.
- `docs/`: public benchmark cards, adapter docs, and claim boundaries.
- `reports/` and `results/`: curated evidence artifacts and generated release checks.

## Reproducibility

Run the quickstart commands from a fresh checkout. For a stricter release check:

```bash
python scripts/check_public_release_ready.py --output results/public_release_audit
python scripts/check_fresh_clone_repro.py --repo-path . --output results/fresh_clone_check
python scripts/check_public_release_package.py --output results/public_release_package_check
```

## Citation

Use the metadata in `CITATION.cff`.

## License

This repository is released under the MIT License. See `LICENSE`.

## Compatibility Status Blocks

The sections below are retained so existing evidence-sync checks can verify the historical benchmark status. They are not stronger public claims.

<!-- SCS_CURRENT_STATUS_START -->
## Current Evidence Status

**Current allowed claim:** A weak but positive low-coverage result under the frozen protocol.

**Expansion status:** Expansion is currently blocked.

**Controlling gates:**
- Practical utility gate: `NARROW_TO_WEAK_LOW_COVERAGE_CLAIM`
- Repair signal role gate: `MARK_REPAIR_DIAGNOSTIC_ONLY_FOR_CSTR`

**What is supported:**
- TwoTank: calibrated low-coverage refusal has a practically meaningful positive effect.
- CSTR: calibrated low-coverage refusal has a positive but practically weak effect.
- `repair_amount` is correct as a bounds/projection signal but diagnostic-only for CSTR.
- `invariant_residual` is much more informative than repair on CSTR.

**What is not supported:**
- strong general selective simulation
- high-coverage reliability
- safety certification
- product readiness
- autonomous control
- plant-wide digital twin claims
- RSSM or third-system evidence
<!-- SCS_CURRENT_STATUS_END -->

<!-- SCS_USABILITY_START -->
## Run the Current Status Demo

```bash
python scripts/run_current_status_demo.py --config configs/status/benchmark_usability_v1_1.yaml --output results/demo
```

## What This Benchmark Tests

It tests refusal/ranking behavior for counterfactual simulator rollouts under intervention shift.

## What This Benchmark Does Not Test

It does not test simulator safety, plant-wide deployment, RSSM evidence, third-system evidence, autonomous control, or high-coverage reliability.

## Add Your Own Model

Implement the adapter described in `docs/custom_model_adapter.md`, inspect `examples/custom_model_example.py`, and run:

```bash
python examples/custom_model_example.py --output results/custom_model_example
```

## Local Model Comparison

```bash
python scripts/compare_models.py --config configs/experiments/calibrated_two_tank.yaml --models hold_last linear_narx mlp_state_space --output results/model_comparison
```

## Current Evidence Status

A weak but positive low-coverage refusal benchmark under a frozen protocol. TwoTank is stronger than CSTR. `repair_amount` is diagnostic-only for CSTR; `invariant_residual` is informative for CSTR.

## Claim Boundaries

This usability release does not change the scientific claim. It does not add RSSM, third-system evidence, product API, frontend, or deployment work.
<!-- SCS_USABILITY_END -->
