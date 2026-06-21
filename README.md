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

The calibrated refusal-judge milestone replaces the failed `combined_linear` claim with a narrower low-coverage claim. Current calibrated evidence status:

- single calibrated TwoTank run: `SUPPORTED_LOW_COVERAGE`
- calibrated seed sweep: `ROBUST_LOW_COVERAGE`
- threshold/coverage stress: `ROBUST_LOW_COVERAGE_ONLY`
- calibrated decision gate: `PROCEED_TO_CSTR`
- CSTR frozen-protocol sanity: `VALID_CSTR_BENCHMARK`
- single calibrated CSTR run: `SUPPORTED_LOW_COVERAGE`
- calibrated CSTR seed sweep: `ROBUST_LOW_COVERAGE`
- calibrated CSTR threshold/coverage stress: `ROBUST_LOW_COVERAGE_ONLY`
- multi-system calibrated gate: `TWO_SYSTEM_LOW_COVERAGE_SUPPORTED`

The multi-system gate allows only the bounded claim stated in `reports/multi_system_calibrated_decision_gate.md`. RSSM, product/platform work, and any frontend/API expansion remain out of scope.

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
python scripts/verify_calibrated_judge_preconditions.py
python scripts/run_calibrated_judge.py --config configs/experiments/calibrated_two_tank.yaml --output results/calibrated_two_tank
python scripts/run_calibrated_seed_sweep.py --config configs/experiments/calibrated_two_tank.yaml --seeds 0 1 2 3 4 5 6 7 8 9 --output results/calibrated_seed_sweep_two_tank
python scripts/run_calibrated_stress.py --config configs/experiments/calibrated_two_tank.yaml --thresholds 0.05 0.10 0.15 0.20 0.30 0.50 --coverages 0.05 0.10 0.20 0.40 0.60 0.80 1.00 --seeds 0 1 2 3 4 --output results/calibrated_stress_two_tank
python scripts/make_calibrated_judge_decision_gate.py --single-run results/calibrated_two_tank/calibrated_judge_summary.json --seed-sweep results/calibrated_seed_sweep_two_tank/seed_sweep_calibrated_summary.json --stress results/calibrated_stress_two_tank/stress_summary.json --output reports/calibrated_judge_decision_gate.md
python scripts/verify_cstr_preconditions.py --protocol docs/calibrated_protocol_lock_v1.md --output results/cstr_preconditions
python scripts/run_cstr_sanity.py --config configs/experiments/calibrated_cstr.yaml --output results/cstr_sanity
python scripts/run_calibrated_judge.py --config configs/experiments/calibrated_cstr.yaml --output results/calibrated_cstr
python scripts/run_calibrated_seed_sweep.py --config configs/experiments/calibrated_cstr.yaml --seeds 0 1 2 3 4 5 6 7 8 9 --output results/calibrated_seed_sweep_cstr
python scripts/run_calibrated_stress.py --config configs/experiments/calibrated_cstr.yaml --thresholds 0.05 0.10 0.15 0.20 0.30 0.50 --coverages 0.05 0.10 0.20 0.40 0.60 0.80 1.00 --seeds 0 1 2 3 4 --output results/calibrated_stress_cstr
python scripts/make_multi_system_calibrated_decision_gate.py --twotank-single results/calibrated_two_tank/calibrated_judge_summary.json --twotank-seed results/calibrated_seed_sweep_two_tank/seed_sweep_calibrated_summary.json --twotank-stress results/calibrated_stress_two_tank/stress_summary.json --cstr-sanity results/cstr_sanity/cstr_label_checks.json --cstr-single results/calibrated_cstr/calibrated_judge_summary.json --cstr-seed results/calibrated_seed_sweep_cstr/seed_sweep_calibrated_summary.json --cstr-stress results/calibrated_stress_cstr/stress_summary.json --output reports/multi_system_calibrated_decision_gate.md
```
