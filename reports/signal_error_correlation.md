# Signal-Error Correlation Report

## Question

Do validator signals predict actual model failure?

## Signals analyzed

support_distance, uncertainty_score, disagreement_score, invariant_residual, repair_amount, combined_linear_score

## Targets analyzed

rmse, mae, max_abs_error, final_state_error, event_error, bad_rmse_label, bad_event_label

## Best signals for RMSE failure

| signal | auroc | auprc | spearman | n |
| --- | ---: | ---: | ---: | ---: |
| uncertainty_score | 1.000000 | 1.000000 | 0.854191 | 48 |
| invariant_residual | 1.000000 | 1.000000 | 0.854446 | 48 |
| repair_amount | 1.000000 | 1.000000 | 0.872575 | 48 |
| combined_linear_score | 0.971429 | 0.979663 | 0.805292 | 48 |
| support_distance | 0.960714 | 0.970875 | 0.787675 | 48 |
| disagreement_score | 0.917857 | 0.942355 | 0.714247 | 48 |

## Best signals for event failure

| signal | auroc | auprc | spearman | n |
| --- | ---: | ---: | ---: | ---: |
| support_distance | nan | nan | nan | 0 |
| uncertainty_score | nan | nan | nan | 0 |
| disagreement_score | nan | nan | nan | 0 |
| invariant_residual | nan | nan | nan | 0 |
| repair_amount | nan | nan | nan | 0 |
| combined_linear_score | nan | nan | nan | 0 |

## Signals that are near random

none

## Signals that are negatively correlated

none

## Interpretation

Best RMSE-failure AUROC was 1.000000. Event labels are unavailable in current v0 artifacts, so event AUROC/AUPRC are not fabricated.

## Verdict

USEFUL_SIGNALS_FOUND
