# Multi-System Calibrated Decision Gate

## Starting point

TwoTank calibrated low-coverage claim passed. CSTR is tested as frozen-protocol replication.

## Protocol lock

docs/calibrated_protocol_lock_v1.md

## TwoTank evidence

single: SUPPORTED_LOW_COVERAGE
seed: ROBUST_LOW_COVERAGE
stress: ROBUST_LOW_COVERAGE_ONLY

## CSTR sanity

VALID_CSTR_BENCHMARK

## CSTR single-run result

SUPPORTED_LOW_COVERAGE

## CSTR seed-sweep result

ROBUST_LOW_COVERAGE

## CSTR threshold/coverage stress result

ROBUST_LOW_COVERAGE_ONLY

## Leakage status

False

## Decision

TWO_SYSTEM_LOW_COVERAGE_SUPPORTED

## Allowed claims

A calibrated low-coverage refusal result replicated on TwoTank and CSTR under the frozen protocol.

## Forbidden claims

product readiness, safety certification, autonomous control, broad plant-wide validity

## Allowed next actions

write up the bounded evidence, inspect failure modes, only then consider separately gated expansion

## Forbidden next actions

RSSM evidence in this milestone, heat_exchanger evidence in this milestone, API/frontend/product work

## Explanation

Decision follows the frozen multi-system gate rules. Oracle remains diagnostic only.
