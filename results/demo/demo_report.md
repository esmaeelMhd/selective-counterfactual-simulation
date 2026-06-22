# Current Status Demo Report

## What this demo does

Runs a lightweight TwoTank local benchmark path with the built-in models and writes a small risk-coverage result table.

## What this demo does not prove

This demo is not the full evidence chain.
The current supported claim remains weak and low-coverage only.

## Current allowed claim

A weak but positive low-coverage refusal benchmark under a frozen protocol.

## Demo result table

| system_id | coverage | baseline_judge | calibrated_judge | baseline_far | calibrated_far | absolute_margin | claim_scope | is_demo |
| --- | ---: | --- | --- | ---: | ---: | ---: | --- | ---: |
| two_tank | 0.050000 | support_only_demo | combined_linear_demo | 0.000000 | 0.000000 | 0.000000 | demo_only_not_full_evidence | True |
| two_tank | 0.100000 | support_only_demo | combined_linear_demo | 0.000000 | 0.000000 | 0.000000 | demo_only_not_full_evidence | True |

## How to reproduce

```bash
python scripts/run_current_status_demo.py --config configs/status/benchmark_usability_v1_1.yaml --output results/demo
```

## Where to find full evidence

See `results/current_status/evidence_manifest/current_evidence_manifest.json` and `reports/current_status_decision_gate.md`.

## Non-claims

This is not safety certification, product readiness, high-coverage reliability, autonomous control, RSSM evidence, or third-system evidence.
