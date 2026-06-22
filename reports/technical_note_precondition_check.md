# Technical Note Package Preconditions

## Working tree

Dirty: False

## Controlling gates

Current status gate: CURRENT_STATUS_SYNCED

Practical utility gate: NARROW_TO_WEAK_LOW_COVERAGE_CLAIM

Repair signal role gate: MARK_REPAIR_DIAGNOSTIC_ONLY_FOR_CSTR

Allowed next action: MAINTAIN_REPO_AS_WEAK_POSITIVE_BENCHMARK

## Expansion status

Expansion allowed: False

## Required source artifacts

| name | path | exists |
| --- | --- | ---: |
| current_manifest | results/current_status/evidence_manifest/current_evidence_manifest.json | True |
| twotank_low_coverage | results/calibrated_two_tank/low_coverage_summary.csv | True |
| cstr_low_coverage | results/calibrated_cstr/low_coverage_summary.csv | True |
| effect_size_summary | results/effect_size_audit/effect_size/effect_size_summary.json | True |
| cstr_weakness_diagnosis | reports/cstr_weakness_diagnosis.md | True |
| repair_signal_role_decision | reports/repair_signal_role_decision_gate.md | True |
| signal_semantics_registry | results/current_status/evidence_manifest/current_evidence_manifest.json | True |
| smoke_model_metrics | results/smoke_two_tank/model_metrics.csv | True |

## Source artifact hash manifest

results/technical_note_package/preconditions/source_artifact_hashes.json

## Forbidden dependency scan

Old repo runtime import hits: none

Path hack hits: none

Forbidden evidence refs: none

## Verdict

READY_FOR_TECHNICAL_NOTE_PACKAGE
