# TwoTank Intervention Severity

Severity controls change the actual generated actions and disturbances for TwoTank OOD splits.

| severity | action magnitude center | action step | inflow spike | combined action center | combined inflow spike | demand drop | pump factor |
|---|---:|---:|---:|---:|---:|---:|---:|
| low | 1.12 | 0.35 | 0.35 | 1.15 | 0.30 | 0.08 | 0.75 |
| medium | 1.55 | 0.85 | 1.15 | 1.65 | 0.95 | 0.20 | 0.45 |
| high | 2.05 | 1.20 | 1.85 | 2.15 | 1.65 | 0.32 | 0.25 |
| extreme | 2.42 | 1.55 | 2.65 | 2.45 | 2.35 | 0.40 | 0.12 |

Affected splits:
- `ood_action_magnitude`
- `ood_inflow_spike`
- `ood_combined`
- `pump_degradation`

