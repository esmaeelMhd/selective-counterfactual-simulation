# Repair-Signal Semantics Preconditions

## Current controlling decision

NARROW_TO_WEAK_LOW_COVERAGE_CLAIM

## Current CSTR weakness diagnosis

REPAIR_SIGNAL_BLIND_SPOT; recommended next action: FIX_REPAIR_SIGNAL

## Expansion status

Expansion allowed: False

## Working tree status

Dirty: yes

## Required artifacts

| path | exists |
| --- | ---: |
| results/calibrated_two_tank | True |
| results/calibrated_cstr | True |
| results/cstr_weakness_audit | True |
| reports/practical_utility_decision_gate.md | True |
| reports/cstr_weakness_diagnosis.md | True |
| docs/calibrated_protocol_lock_v1.md | True |
| reports/practical_utility_decision_gate.json | True |
| reports/cstr_weakness_diagnosis.json | True |
| results/calibrated_two_tank/calibration_table.csv | True |
| results/calibrated_two_tank/test_table.csv | True |
| results/calibrated_cstr/calibration_table.csv | True |
| results/calibrated_cstr/test_table.csv | True |

## Protocol lock status

Protocol lock exists: True

## Forbidden dependency scan

Old repo runtime import hits: none
Path hack hits: none
Forbidden evidence refs: none

## Prior-artifact mutation policy

prior calibrated/effect/CSTR weakness evidence directories are read-only for this audit

## Verdict

READY_FOR_REPAIR_SIGNAL_SEMANTICS_AUDIT
