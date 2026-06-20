# Selective Counterfactual Simulation

This repository tests one research question:

> Can a learned or hybrid simulator identify which counterfactual intervention scenarios it can answer reliably and abstain on the rest?

The first milestone is an end-to-end smoke benchmark on a simulated TwoTank system. It trains three lightweight simulator models, evaluates at least six refusal judges, and produces risk-coverage artifacts:

- `results/smoke_two_tank/risk_coverage.csv`
- `results/smoke_two_tank/risk_coverage.png`
- `results/smoke_two_tank/summary.json`
- `reports/smoke_report.md`

The primary metric is false accept rate at fixed coverage. A false accept occurs when a judge accepts a scenario whose simulator prediction is materially wrong under the configured error threshold.

## Current Evidence Status

The v0 evidence audit downgraded the primary claim. `combined_linear` did not robustly beat the strongest simple judge in the claim audit or seed sweep, so it should be treated as an exploratory baseline until v0 is fixed and re-audited.

## Non-goals

This is not a product, service, plant-wide digital twin, control stack, API, frontend, or safety certification workflow. The v0 scope is a runnable benchmark with explicit failures and measured risk-coverage curves.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
python scripts/run_smoke.py
```

## Main Commands

```bash
python scripts/generate_data.py --config configs/experiments/smoke_two_tank.yaml
python scripts/train_model.py --config configs/experiments/smoke_two_tank.yaml --model linear_narx
python scripts/evaluate_selective.py --config configs/experiments/smoke_two_tank.yaml
python scripts/make_report.py --results results/smoke_two_tank
python scripts/run_smoke.py
python scripts/audit_claim.py --results results/smoke_two_tank
python scripts/make_decision_gate.py
python scripts/run_smoke.py --config configs/experiments/smoke_cstr.yaml
python scripts/audit_claim.py --results results/smoke_cstr --report reports/cstr_claim_audit.md
python scripts/make_multi_system_report.py --results results/smoke_two_tank results/smoke_cstr --output reports/multi_system_report.md
```
