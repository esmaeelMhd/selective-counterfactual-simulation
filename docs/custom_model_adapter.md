# Custom Model Adapter

## Purpose

The adapter lets you compare your own simulator model locally without changing benchmark internals or the current evidence claim.

Custom model results are local comparison results only. They are not evidence for the current supported claim.

## Required methods

Implement a class with:

```python
model_id: str

def fit(self, train_batch) -> None:
    ...

def predict_rollout(self, initial_state, actions, disturbances):
    ...
```

The helper base class is `src/scs/models/user_model.py::UserSimulatorModel`.

## Expected shapes

- `initial_state`: `(state_dim,)`
- `actions`: `(horizon, action_dim)`
- `disturbances`: `(horizon, disturbance_dim)`
- return value: `(horizon + 1, state_dim)`

The adapter validates rollout shape and finite values. Shape errors fail loudly.

## Minimal example

See `examples/custom_model_example.py::DampedLinearUserModel`.

For a blank starting point, copy `examples/my_model_template.py`.

```python
class MySimulatorModel:
    model_id = "my_simulator"

    def fit(self, train_batch):
        ...

    def predict_rollout(self, initial_state, actions, disturbances):
        ...
```

## How to run

```bash
python examples/custom_model_example.py --output results/custom_model_example
python scripts/compare_models.py --config configs/experiments/calibrated_two_tank.yaml --models linear_narx mlp_state_space --custom-model examples/custom_model_example.py:DampedLinearUserModel --output results/model_comparison_custom
```

The runnable example writes:

- `results/custom_model_example/custom_model_smoke.json`
- `results/custom_model_example/custom_model_report.md`

## Common errors

- Returning `(horizon, state_dim)` instead of `(horizon + 1, state_dim)`.
- Returning non-finite values.
- Changing the initial state instead of placing it at `rollout[0]`.
- Training on test or evidence artifacts instead of the generated train split.

## What this does not prove

This does not modify the current evidence manifest. It does not support stronger reliability, safety, product, high-coverage, RSSM, or third-system claims.

## Fair comparison rules

- Use the same generated train/test split for every model in a local comparison.
- Report custom results as local-only.
- Do not mix custom model outputs into the frozen evidence claim.
- Keep the current allowed claim weak and low-coverage only.
