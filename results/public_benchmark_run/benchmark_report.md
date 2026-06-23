# Public Benchmark Run Report

## Command

```bash
python scripts/run_benchmark.py --model examples/custom_model_example.py:DampedLinearUserModel --output results/public_benchmark_run
```

## Models evaluated

damped_linear_user

## Systems evaluated

cstr, heat_exchanger

## Badness targets

bad_rmse, bad_event, bad_rmse_or_event

## Risk-coverage summary

| badness_target | judge_id | false_accept_rate |
| --- | --- | ---: |
| bad_event | event_guard_public | 0.069940 |
| bad_event | invariant_only | 0.040179 |
| bad_event | repair_only | 0.069940 |
| bad_event | support_only | 0.052083 |
| bad_rmse | event_guard_public | 0.531845 |
| bad_rmse | invariant_only | 0.319048 |
| bad_rmse | repair_only | 0.531845 |
| bad_rmse | support_only | 0.515476 |
| bad_rmse_or_event | event_guard_public | 0.531845 |
| bad_rmse_or_event | invariant_only | 0.319048 |
| bad_rmse_or_event | repair_only | 0.531845 |
| bad_rmse_or_event | support_only | 0.515476 |

## Event-risk summary

| system_id | model_id | event_mismatch_rate | n_scenarios |
| --- | --- | ---: | ---: |
| cstr | damped_linear_user | 0.000000 | 70 |
| heat_exchanger | damped_linear_user | 0.071429 | 70 |

## Accepted false accepts

Rows written: 99

## What this run proves

This run proves that the selected model can be evaluated through the public benchmark interface and produces risk-coverage artifacts.

## What this run does not prove

This benchmark run does not update the repository's current scientific claim. It does not prove simulator trustworthiness, safety certification, product readiness, or general reliability.
