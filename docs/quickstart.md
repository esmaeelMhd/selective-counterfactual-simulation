# Quickstart

## Install

```bash
pip install -e ".[dev]"
pytest -q
```

## Smoke demo

```bash
python scripts/run_smoke_demo.py --output results/smoke_demo
```

This checks that the benchmark pipeline runs. It is not the full evidence reproduction.

## Main TwoTank reproduction

```bash
python scripts/reproduce_main_twotank_result.py --output results/reproduce_twotank
```

## Local model comparison

```bash
python scripts/compare_models.py --config configs/experiments/calibrated_two_tank.yaml --models hold_last linear_narx mlp_state_space --output results/model_comparison
```

## Custom model comparison

```bash
python scripts/compare_models.py --config configs/experiments/calibrated_two_tank.yaml --models linear_narx mlp_state_space --custom-model examples/custom_model_example.py:DampedLinearUserModel --output results/model_comparison_custom
```

## Claim boundary

The current claim remains weak-positive and low-coverage only.
