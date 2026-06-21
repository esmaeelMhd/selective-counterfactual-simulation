# CSTR Weakness Audit Preconditions

## Current controlling decision

NARROW_TO_WEAK_LOW_COVERAGE_CLAIM

## Expansion status

Expansion allowed: False

## Required artifacts

| path | exists |
| --- | ---: |
| results/calibrated_cstr/calibrated_risk_coverage.csv | True |
| results/calibrated_cstr/test_table.csv | True |
| results/calibrated_cstr/calibrated_judge_summary.json | True |
| results/calibrated_cstr/low_coverage_summary.csv | True |
| results/effect_size_audit/false_accept_forensics/accepted_false_accepts.csv | True |
| results/effect_size_audit/false_accept_forensics/false_accept_tag_counts.csv | True |
| reports/practical_utility_decision_gate.md | True |
| docs/calibrated_protocol_lock_v1.md | True |

## Protocol lock status

Protocol lock exists: True

## Forbidden dependency scan

Old repo runtime import hits: none
Path hack hits: none

## Artifact mutation policy

prior evidence directories are read-only for this audit

## Verdict

READY_FOR_CSTR_WEAKNESS_AUDIT
