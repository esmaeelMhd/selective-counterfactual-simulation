# False-Accept Forensics Tags

These tags are diagnostic labels for accepted bad scenarios in the effect-size audit. They do not change the calibrated refusal protocol.

## LOW_SUPPORT_RISK_BUT_BAD

The scenario was accepted as low risk and its support distance was in the lowest quartile for that system, but its RMSE exceeded the bad threshold.

## LOW_UNCERTAINTY_BUT_BAD

The scenario was accepted as low risk and its uncertainty score was in the lowest quartile for that system, but its RMSE exceeded the bad threshold.

## LOW_DISAGREEMENT_BUT_BAD

The scenario was accepted as low risk and its model disagreement score was in the lowest quartile for that system, but its RMSE exceeded the bad threshold.

## LOW_INVARIANT_RESIDUAL_BUT_BAD

The scenario was accepted as low risk and its invariant residual was in the lowest quartile for that system, but its RMSE exceeded the bad threshold.

## LOW_REPAIR_BUT_BAD

The scenario was accepted as low risk and its repair amount was in the lowest quartile for that system, but its RMSE exceeded the bad threshold.

## MODEL_SPECIFIC_FAILURE

Accepted false accepts are concentrated in the same simulator model.

## SPLIT_SPECIFIC_FAILURE

Accepted false accepts are concentrated in the same split or scenario type.

## NEAR_THRESHOLD_FAILURE

The accepted false accept exceeded the RMSE threshold by no more than 25 percent.

## SEVERE_MISCLASSIFICATION

The accepted false accept had RMSE at least three times the configured bad threshold.
