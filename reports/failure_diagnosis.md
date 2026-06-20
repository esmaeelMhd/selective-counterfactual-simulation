# Failure Diagnosis

## Starting point

The v0 decision gate did not allow expansion. The original combined_linear claim is not supported.

## Evidence summary

| analysis | verdict | key_finding |
| --- | --- | --- |
| gate | READY_FOR_FAILURE_ANALYSIS | BLOCKED |
| signal_error_correlation | USEFUL_SIGNALS_FOUND | Best RMSE-failure AUROC was 0.891903 |
| per_split_failure | GLOBAL_FAILURE | Worst split by mean RMSE: ood_combined |
| threshold_sensitivity | UNSUPPORTED_ACROSS_THRESHOLDS | combined_linear worked at 0 of 6 thresholds |
| coverage_sensitivity | WORKS_AT_LOW_COVERAGE | combined_linear worked at 1 coverage points |
| score_ablation | SIGNAL_PROBLEM | Best ablation/calibration win rate was 0.000000 |
| model_diversity | ORACLE_GAP_SMALL | Mean oracle gap was 0.000278 |
| benchmark_sanity | VALID_BENCHMARK | OOD mean error 0.388781 vs ID mean error 0.171247 |

## Diagnosis

JUDGE_PROBLEM

## Explanation

The diagnosis follows the rule hierarchy in the failure-analysis task. Expansion remains blocked by the v0 gate unless a future minimal test changes the diagnosis.

## What not to do next

- Do not add CSTR, RSSM, or new systems as a way to rescue the failed v0 claim.
- Do not treat oracle_error_rank as a deployable judge.
- Do not call combined_linear supported without rerunning this diagnosis after a real fix.

## Recommended next action

REPLACE_JUDGE

## Concrete next milestone

Run one minimal fix aligned with the diagnosis, then rerun the full failure-analysis command chain.

## Claims allowed after this analysis

- The current v0 claim is unsupported.
- The listed diagnostics identify the current failure mode for this setup.

## Claims still forbidden

- combined_linear is robustly better than the strongest simple judge.
- Expansion results validate the original claim.
