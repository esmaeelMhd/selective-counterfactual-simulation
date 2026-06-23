# v2 Comparator Taxonomy

## Purpose

This document separates deployable comparator baselines from diagnostic upper-bound comparisons for the v2 comparator fairness audit. It exists to prevent the row-wise strongest-baseline envelope from being interpreted as a deployable method.

## Comparator 1: row-wise strongest-baseline envelope

The row-wise strongest-baseline envelope is a diagnostic-only comparator. For each frozen test row, it uses the already-recorded lowest false-accept rate among baseline judges at the same system, seed, model, badness target, threshold, and coverage. It uses test outcomes row-wise and is therefore an upper-bound comparator, not a deployable baseline.

## Comparator 2: global calibration-selected fixed baseline

The global calibration-selected fixed baseline selects one baseline judge per seed and coverage using calibration rows only. The selected judge is then evaluated on frozen test rows across systems, targets, thresholds, and models.

## Comparator 3: per-system calibration-selected fixed baseline

The per-system calibration-selected fixed baseline selects one baseline judge per seed, system, and coverage using calibration rows only. The selected judge is then evaluated on frozen test rows for that system.

## Comparator 4: per-system-target calibration-selected fixed baseline

The per-system-target calibration-selected fixed baseline selects one baseline judge per seed, system, badness target, and coverage using calibration rows only. The selected judge is then evaluated on frozen test rows for that system and target.

## Comparator 5: current primary calibrated judge

The current primary calibrated judge is `calibration_selected_candidate_ranker`. It is the method whose v2 underperformance is being audited. Its frozen test false-accept rates are read from the v2 artifacts.

## Comparator 6: best calibrated-family member selected on calibration

The best calibrated-family member is selected from the calibrated-family judges using calibration rows only, with the same deterministic tie-breakers used for deployable baseline selection. It is evaluated on frozen test rows as a calibrated-family diagnostic, not as a new method.

## Deployable vs diagnostic comparators

Deployable comparators are selected without test labels and can be specified before test evaluation. Diagnostic comparators use test outcomes or oracle information and cannot be deployed without leakage.

Deployable comparators:
- global calibration-selected fixed baseline
- per-system calibration-selected fixed baseline
- per-system-target calibration-selected fixed baseline
- current primary calibrated judge
- best calibrated-family member selected on calibration

Diagnostic comparators:
- row-wise strongest-baseline envelope
- oracle error rank

## What uses test labels

The row-wise strongest-baseline envelope uses test false-accept outcomes to choose the best baseline row by row. Oracle error rank uses test error labels directly. Both are diagnostic only.

## What does not use test labels

The global, per-system, and per-system-target fixed baselines use calibration rows only. The best calibrated-family member selected on calibration also uses calibration rows only.

## Allowed interpretation

It is allowed to say that the row-wise envelope is stricter than deployable fixed baselines. It is allowed to report whether the calibrated candidate fails or improves against fair deployable baselines under the frozen v2 data.

## Forbidden interpretation

Do not call the row-wise strongest-baseline envelope deployable. Do not use it as evidence that a practical baseline can adapt per test row. Do not upgrade any calibrated-refusal claim unless the comparator fairness decision gate explicitly permits it.
