# Benchmark API Contract

## Purpose

This contract defines the minimal interface for plugging an external simulator into the public benchmark.

## Minimal model interface

```python
class MySimulator:
    model_id: str

    def fit(self, train_batch) -> None:
        ...

    def predict_rollout(self, initial_state, actions, disturbances):
        ...
```

## Required method: fit

`fit(train_batch)` receives a `TrajectoryBatch` with states, actions, disturbances, scenario labels, split name, and system id. Deterministic models may store fitted parameters and return `None`.

## Required method: predict_rollout

`predict_rollout(initial_state, actions, disturbances)` returns a single open-loop trajectory prediction.

## Expected input shapes

- `initial_state`: `(state_dim,)`
- `actions`: `(horizon, action_dim)`
- `disturbances`: `(horizon, disturbance_dim)`

## Expected output shapes

The output must be finite and have shape `(horizon + 1, state_dim)`. The first row should correspond to the initial predicted state.

## Batch conventions

The benchmark calls `predict_rollout` one scenario at a time. It handles batching externally.

## Handling deterministic models

Deterministic models only need `fit` and `predict_rollout`. The public benchmark does not require sampled rollouts.

## Handling stochastic/ensemble models

Stochastic or ensemble models should make `predict_rollout` deterministic for a fixed fitted model and input, or document their internal random seed behavior.

## Error handling

The benchmark fails loudly if `fit` or `predict_rollout` is missing, if output shapes are wrong, or if predictions contain NaN or infinite values.

## Fair comparison rules

Custom model runs are local benchmark runs only. They do not update the repository's current scientific claim or frozen v2 evidence.

## Example model

See `examples/custom_model_example.py:DampedLinearUserModel`.

## Non-goals

This API is not a product API, service API, cloud interface, or safety certification interface.
