# Task: Selective Counterfactual Simulation Refusal

## Task summary

Given a learned dynamical simulator and counterfactual intervention scenarios, rank scenarios by answerability and accept only the lowest-risk fraction.

## Input

The input is a simulator model plus synthetic system trajectories containing states, actions, disturbances, scenario labels, and train/calibration/test splits.

## Output

The output is a risk-coverage table, model metrics, event metrics, accepted false accepts, and a benchmark report.

## Metric: false accept rate at coverage

False accept rate is the fraction of accepted scenarios that are bad under the selected badness target.

## Badness targets

- `bad_rmse`: trajectory RMSE exceeds the configured public threshold.
- `bad_event`: predicted event status mismatches true event status.
- `bad_rmse_or_event`: either RMSE is bad or event status is bad.

## Coverage definition

Coverage is the fraction of scenarios accepted after sorting by risk score from lowest risk to highest risk.

## False accept definition

A false accept occurs when a scenario is accepted and the simulator prediction is bad under the active badness target.

## Systems

The public command uses a small benchmark subset of existing systems. It does not add systems or mutate frozen v2 evidence.

## Models

Users may run built-in models or a custom model implementing the benchmark API contract.

## Baselines

The public command includes simple risk rankings based on existing support, invariant, repair, and event-guard scores.

## Event-risk caveat

The current v2 finding is that calibrated refusal is target-dependent and not reliable for event-risk.

## How to run your model

```bash
python scripts/run_benchmark.py --model examples/custom_model_example.py:DampedLinearUserModel --output results/public_benchmark_run
```

## Fair comparison rules

Public benchmark runs are local comparisons. They do not update current evidence manifests, decision gates, or allowed claims.

## Non-goals

This benchmark does not provide safety certification, product readiness, general simulator reliability, high-coverage reliability, or a validated digital twin.
