# Selective Counterfactual Simulation Benchmark

## What it is

A Python research benchmark for testing when learned simulators should answer or refuse counterfactual intervention scenarios.

## What problem it tests

The benchmark tests selective prediction under intervention shift: accept low-risk scenarios and abstain on scenarios likely to produce materially wrong rollouts.

## What I built

I built synthetic TwoTank and CSTR systems, data generation, three simulator models, refusal signals, calibrated judge selection, risk-coverage metrics, evidence audits, and reproducible reports.

## Key result

A weak but positive low-coverage refusal benchmark under a frozen protocol. TwoTank is practically meaningful; CSTR is positive but weak.

## Important negative findings

The original `combined_linear` claim was downgraded. `repair_amount` is diagnostic-only for CSTR because it misses within-bound dynamic errors. `invariant_residual` is much more informative for CSTR.

## What it does not claim

This is not safety certification, product readiness, autonomous control, plant-wide simulation, high-coverage reliability, RSSM evidence, or third-system evidence.

## Reproducibility

Run `pip install -e ".[dev]"`, `pytest -q`, and `python scripts/run_smoke.py`. The current status gate is `current_evidence_status_v1` with expansion blocked.
