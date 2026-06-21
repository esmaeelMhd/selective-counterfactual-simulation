# Repair Signal Role Decision Gate

## Starting point

CSTR weakness diagnosis identified REPAIR_SIGNAL_BLIND_SPOT.

## Controlled repair validation

REPAIR_CORRECT_BUT_CSTR_IRRELEVANT

## Repair vs invariant comparison

INVARIANT_DOMINATES_REPAIR

## Signal-set ablation

NO_REPAIR_NO_BENEFIT

## Seed robustness

NO_SEED_STABLE_BENEFIT

## Decision

MARK_REPAIR_DIAGNOSTIC_ONLY_FOR_CSTR

## Allowed next action

UPDATE_SIGNAL_SEMANTICS_ONLY

## Forbidden next actions

ADD_RSSM, ADD_THIRD_SYSTEM, ADD_NEW_JUDGE_FAMILY, MUTATE_FROZEN_PROTOCOL

## Allowed claim after this gate

A weak but positive low-coverage result under the frozen protocol, with repair_amount treated according to the role decision gate.

## Forbidden claims

general reliability, safety certification, product readiness, strong two-system support
