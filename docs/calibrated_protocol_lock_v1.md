# Calibrated Protocol Lock v1

## Source checkpoint

Commit: a7d0ce3
Tag: v1-calibrated-refusal-judge-improved

## Purpose

This document freezes the calibrated low-coverage refusal protocol before CSTR replication evidence is generated. CSTR is replication evidence only; it is not a design playground and it does not establish a general claim by itself.

## Candidate judges

- best_single_signal_selected_on_calibration
- rank_normalized_linear
- logistic_calibrated_judge
- isotonic_calibrated_judge
- quantile_rule_judge
- conservative_low_coverage_judge
- calibration_selected_candidate_ranker

## Real baseline judges

- support_only
- uncertainty_only
- disagreement_only
- invariant_only
- repair_only
- combined_linear
- random_baseline

## Diagnostic-only judges

- oracle_error_rank

## Signal columns

- support_distance
- uncertainty_score
- disagreement_score
- invariant_residual
- repair_amount

## Models

- hold_last
- linear_narx
- mlp_state_space

## Data roles

- model_train: simulator model fitting only.
- judge_calibration: calibrated judge selection, orientation, weights, hyperparameters, and calibration-only labels.
- judge_test: final fixed-judge risk-coverage evaluation only.

## Calibration rules

Simulator models fit on model_train only. Calibrated judges may fit or select parameters on judge_calibration rows only. The best single-signal baseline is selected using calibration labels only. Oracle error ranking is diagnostic only and cannot support a claim.

## Primary coverages

- 0.05
- 0.10

## Full coverage grid

- 0.05
- 0.10
- 0.20
- 0.40
- 0.60
- 0.80
- 1.00

## Bad-threshold grid

- 0.05
- 0.10
- 0.15
- 0.20
- 0.30
- 0.50

## Seed-sweep rules

Run ten seeds, 0 through 9. ROBUST_LOW_COVERAGE requires low-coverage calibrated wins in at least seven of ten seeds. UNSTABLE is four to six wins. NO_ROBUST_IMPROVEMENT is three or fewer wins. Leakage in any seed invalidates the result.

## Stress-test rules

Run thresholds 0.05, 0.10, 0.15, 0.20, 0.30, and 0.50 across coverages 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, and 1.00 on seeds 0 through 4. Degenerate thresholds are reported and cannot be counted as wins.

## CSTR decision rules

SUPPORTED_LOW_COVERAGE requires a calibrated judge to beat best_single_signal_selected_on_calibration and combined_linear at coverage 0.05 or 0.10 with no leakage and a valid CSTR benchmark. MIXED means wins exist but are not consistent. NO_IMPROVEMENT_OVER_SINGLE_SIGNAL means calibrated judges do not beat the calibration-selected single-signal baseline. INVALID_DUE_TO_LEAKAGE and INVALID_BENCHMARK override all positive outcomes.

## Multi-system decision rules

TWO_SYSTEM_LOW_COVERAGE_SUPPORTED requires TwoTank single, seed, and stress verdicts to remain supported and CSTR sanity, single, seed, and stress verdicts to pass their corresponding rules with no leakage. TWOTANK_ONLY_SUPPORTED means TwoTank remains robust but CSTR has no robust improvement. MIXED_SYSTEM_EVIDENCE covers CSTR mixed, unstable, or threshold-dependent outcomes. NO_GENERALIZATION covers valid CSTR benchmark failure. INVALID_CSTR_BENCHMARK and INVALID_DUE_TO_LEAKAGE override all other decisions.

## Forbidden changes

- RSSM evidence
- heat_exchanger evidence
- new systems beyond CSTR
- new calibrated judge candidates
- new refusal signals
- new model families
- changed primary coverages
- changed coverage grid
- changed threshold grid
- changed seed-sweep rules
- changed stress-test rules
- changed decision-gate rules
- changed success criteria after seeing CSTR results
- API, frontend, dashboard, database, plant compiler, MPC, RL, product workflow, LLM workflow, broad simulator architecture, or large refactor

## Allowed CSTR-specific changes

- state names
- action names
- disturbance names
- physical parameters
- scenario types
- event thresholds
- CSTR data generation needed to instantiate the frozen roles and intervention splits

## Leakage rules

No test labels, test errors, or test false-accept outcomes may be used to choose selected signal, score orientation, judge weights, hyperparameters, candidate judge, thresholds, or decision rules. Calibration/test split overlap invalidates the result. Oracle remains diagnostic only.

## Report verdicts

- READY_FOR_CSTR_SANITY
- NOT_READY
- VALID_CSTR_BENCHMARK
- WEAK_CSTR_BENCHMARK
- INVALID_CSTR_BENCHMARK
- SUPPORTED_LOW_COVERAGE
- MIXED
- NO_IMPROVEMENT_OVER_SINGLE_SIGNAL
- INVALID_DUE_TO_LEAKAGE
- ROBUST_LOW_COVERAGE
- UNSTABLE
- NO_ROBUST_IMPROVEMENT
- ROBUST_LOW_COVERAGE_ONLY
- THRESHOLD_DEPENDENT
- NO_STABLE_REGION
- TWO_SYSTEM_LOW_COVERAGE_SUPPORTED
- TWOTANK_ONLY_SUPPORTED
- MIXED_SYSTEM_EVIDENCE
- NO_GENERALIZATION
