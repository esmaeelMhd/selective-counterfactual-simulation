# v2 Comparator Claim Audit

| Claim | Status | Evidence | Allowed wording |
|---|---|---|---|
| calibrated method fails against row-wise envelope | diagnostic-only evidence | comparator fairness evaluation | The calibrated candidate is compared against a diagnostic upper-bound envelope. |
| row-wise envelope is deployable | forbidden | comparator taxonomy | The row-wise envelope is diagnostic only, not deployable. |
| calibrated method fails against fair fixed baseline | decision-gated | comparator decision gate | Calibrated refusal is target-dependent and not reliable for event-risk. |
| calibrated method helps RMSE but not events | target-dependent if supported by statistics | RMSE and event target results | Only state the target-specific result reported by the audit. |
| conformal baseline dominates | baseline-specific observation only | selection tables and frozen risk coverage | A named baseline may dominate within this benchmark; do not generalize. |
| event-risk is main failure mode | allowed if event rows show worsening | event-risk target result | Event-risk remains a separately reported failure mode. |
| benchmark exposes method failure | allowed when decision says failure or target dependence | decision gate | Calibrated refusal is target-dependent and not reliable for event-risk. |
