# Effect-Size Audit Preconditions

## Required artifacts

| artifact | exists |
| --- | ---: |
| results/calibrated_two_tank/calibrated_judge_summary.json | True |
| results/calibrated_two_tank/calibrated_risk_coverage.csv | True |
| results/calibrated_two_tank/test_table.csv | True |
| results/calibrated_seed_sweep_two_tank/seed_sweep_calibrated_summary.json | True |
| results/calibrated_seed_sweep_two_tank/calibrated_risk_coverage_all.csv | True |
| results/calibrated_stress_two_tank/stress_summary.json | True |
| results/calibrated_cstr/calibrated_judge_summary.json | True |
| results/calibrated_cstr/calibrated_risk_coverage.csv | True |
| results/calibrated_cstr/test_table.csv | True |
| results/calibrated_seed_sweep_cstr/seed_sweep_calibrated_summary.json | True |
| results/calibrated_seed_sweep_cstr/calibrated_risk_coverage_all.csv | True |
| results/calibrated_stress_cstr/stress_summary.json | True |
| results/cstr_sanity/cstr_label_checks.json | True |
| reports/multi_system_calibrated_decision_gate.md | True |
| reports/multi_system_calibrated_decision_gate.json | True |
| docs/calibrated_protocol_lock_v1.md | True |

## Prior verdicts

| artifact | verdict |
| --- | --- |
| two_tank_single | SUPPORTED_LOW_COVERAGE |
| two_tank_seed | ROBUST_LOW_COVERAGE |
| two_tank_stress | ROBUST_LOW_COVERAGE_ONLY |
| cstr_sanity | VALID_CSTR_BENCHMARK |
| cstr_single | SUPPORTED_LOW_COVERAGE |
| cstr_seed | ROBUST_LOW_COVERAGE |
| cstr_stress | ROBUST_LOW_COVERAGE_ONLY |
| multi_gate | TWO_SYSTEM_LOW_COVERAGE_SUPPORTED |

## Leakage status

False

## Protocol lock status

True

## Practical threshold config

minimum_absolute_far_reduction: 0.05
minimum_relative_far_reduction: 0.1
confidence_level: 0.95

## Verdict

READY_FOR_EFFECT_SIZE_AUDIT
