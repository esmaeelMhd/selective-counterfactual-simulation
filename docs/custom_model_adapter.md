# Custom Model Adapter

## Minimal interface

Implement a Python class with:

```python
model_id: str

def fit(self, train_batch) -> None:
    ...

def predict_rollout(self, initial_state, actions, disturbances):
    ...
```

`predict_rollout` must return a finite array with shape `(horizon + 1, state_dim)`.

## Example

See `examples/custom_model_example.py:DampedLinearUserModel` for a runnable adapter.

## How to run

```bash
python scripts/run_benchmark.py --model examples/custom_model_example.py:DampedLinearUserModel --output results/public_benchmark_run
```

You can replace the model spec with `path/to/file.py:ClassName`.

## Expected outputs

- `risk_coverage.csv`
- `model_metrics.csv`
- `event_metrics.csv`
- `accepted_false_accepts.csv`
- `benchmark_summary.json`
- `benchmark_report.md`

## Common errors

- returning `(horizon, state_dim)` instead of `(horizon + 1, state_dim)`;
- returning NaN or infinite values;
- changing the first row instead of returning the initial state at `rollout[0]`;
- using evidence artifacts as training data.

## Validation checks

The benchmark validates required methods, rollout shape, finite values, and command completion. Custom model results are local comparison outputs only and do not update the repository's current evidence claim.
