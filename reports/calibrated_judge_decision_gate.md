# Calibrated Judge Decision Gate

## Starting point

The v0 combined_linear claim was not supported. This gate tests whether calibrated refusal replaces it.

## Single-run result

MIXED

## Seed-sweep result

ROBUST_LOW_COVERAGE

## Threshold/coverage stress result

ROBUST_LOW_COVERAGE_ONLY

## Leakage status

False

## Decision

KEEP_WITH_LOW_COVERAGE_CLAIM

## Allowed next actions

replace calibrated judge and rerun TwoTank evidence

## Forbidden next actions

CSTR evidence, RSSM, new systems, platform/product work

## Explanation

Decision follows the calibrated-judge gate rules. Oracle is diagnostic only and is not used as evidence.
