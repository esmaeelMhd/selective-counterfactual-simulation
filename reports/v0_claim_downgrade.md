# v0 Claim Downgrade

## Decision

DOWNGRADE_CLAIM

## Previous claim

A combined refusal judge reduces false acceptance compared with simple uncertainty-only, support-only, and disagreement-only judges.

## Downgraded claim

The current v0 evidence does not support the claim that combined_linear robustly reduces false accepts against the strongest simple judge. combined_linear remains an exploratory baseline, not a supported method.

## Evidence

| check | result |
|---|---|
| freeze result | ACCEPTED |
| claim audit result | NOT_SUPPORTED |
| claim overall win rate | 0.0 |
| seed robustness result | NOT_SUPPORTED |
| seed mean win rate | 0.0 |
| severity sweep result | MEANINGFUL |
| decision gate | KILL_OR_DOWNGRADE_CLAIM |

## Blocked actions

- Do not use current v0 as positive support for the primary claim.
- Do not proceed to additional systems as claim-support evidence until v0 is fixed.
- Do not add RSSM as a distraction from the failed refusal-ranking evidence.

## Required next action

Fix v0 first: redesign or calibrate the combined judge, rerun claim audit, seed sweep, severity sweep, and decision gate before treating expansion results as evidence for the primary claim.
