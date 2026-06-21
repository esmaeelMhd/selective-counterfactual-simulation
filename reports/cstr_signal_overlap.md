# CSTR Signal Overlap and Separability Audit

## Question

Can current signals separate accepted-good from accepted-bad CSTR scenarios?

## Accepted-good vs accepted-bad separability

| signal | auroc | cohens_d | overlap_coefficient | verdict |
| --- | ---: | ---: | ---: | --- |
| support_distance | 0.652896 | 0.676255 | 0.726107 | MIXED |
| uncertainty_score | 0.097060 | -2.266631 | 0.306527 | FAILS |
| disagreement_score | 0.662393 | 0.757073 | 0.634033 | MIXED |
| invariant_residual | 0.938531 | 1.119443 | 0.186869 | SEPARATES |
| repair_amount | 0.500000 | 0.000000 | 1.000000 | FAILS |
| risk_score | 0.470042 | -0.282111 | 0.386169 | FAILS |

## Signals that separate

invariant_residual

## Signals that fail

uncertainty_score, repair_amount, risk_score

## Interpretation

best deployable accepted-region signal is invariant_residual with AUROC 0.938531

## Verdict

SIGNALS_SEPARATE_ACCEPTED_FAILURES
