# v2 Scientific Decision Gate

## Starting v1 claim

A weak but positive low-coverage result under the frozen protocol.

## v2 protocol status

Protocol hash: d05128c44dde09e3a82fbc3ec5a723242eba30057da6f88be1c2577fdff9503a

Protocol contains forbidden post-result tuning rule: True

## Valid systems

two_tank, cstr, heat_exchanger

## Models evaluated

hold_last, linear_narx, mlp_state_space, ensemble_mlp, gradient_boosted_narx

## Badness targets

bad_rmse, bad_event, bad_rmse_or_event

## Statistical evidence

Statistical verdict: NO_ROBUST_EFFECT

Positive systems: []

Practical-threshold systems: []

CI-positive systems: []

## Event-risk evidence

Event-risk worsening: True

## Decision

NO_METHOD_CLAIM_BENCHMARK_ONLY

## Allowed claim

This repository is a benchmark only; v2 does not support a calibrated-refusal method claim.

## Forbidden claims

- safety certification
- trusted simulator
- validated digital twin
- general simulator reliability
- high-coverage reliability
- product readiness

## Recommended next action

Keep v2 evidence separate from v1 unless a future protocol explicitly permits a claim update.
