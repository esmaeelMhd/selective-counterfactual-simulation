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
| combined_linear_score | 0.891903 | 0.962067 | 0.600351 | 600 |
| support_distance | 0.877131 | 0.943126 | 0.577727 | 600 |
| invariant_residual | 0.858437 | 0.950888 | 0.549085 | 600 |
| disagreement_score | 0.776648 | 0.903705 | 0.423797 | 600 |
| uncertainty_score | 0.284908 | 0.632190 | -0.330276 | 600 |
| repair_amount | nan | nan | nan | 600 |

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

uncertainty_score

## Interpretation

Best RMSE-failure AUROC was 0.891903. Event labels are unavailable in current v0 artifacts, so event AUROC/AUPRC are not fabricated.

## Verdict

USEFUL_SIGNALS_FOUND
