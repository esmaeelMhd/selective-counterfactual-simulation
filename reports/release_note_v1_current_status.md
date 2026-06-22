# Release Note: Weak-Positive Low-Coverage Refusal Benchmark

## What changed

This release packages the current benchmark status into a limitations-first technical note, one-page summary, portfolio summary, claim audit table, reproducibility card, and release manifest.

## Current allowed claim

A weak but positive low-coverage refusal benchmark under a frozen protocol.

## Main evidence

TwoTank is practically meaningful at low coverage. CSTR is positive but weak. The package uses the frozen current-status evidence and does not add new experiments.

## Negative findings

The original `combined_linear` claim was downgraded. repair_amount is diagnostic-only for CSTR. invariant_residual is the more informative CSTR signal.

## Limitations

- TwoTank is practically meaningful; CSTR is positive but weak.
- The result is low-coverage only.
- repair_amount is diagnostic-only for CSTR.
- invariant_residual is much more informative for CSTR.
- Expansion is blocked.
- No safety, product, or general reliability claim is supported.

## Reproducibility

- `pip install -e ".[dev]"`
- `pytest -q`
- `python scripts/run_smoke.py`
- `python scripts/check_technical_note_package.py --config configs/status/technical_note_package.yaml --manifest results/technical_note_package/package_manifest.json`

## What not to do next

Expansion is blocked. Maintain the repository as a weak-positive benchmark; do not claim safety certification, product readiness, high-coverage reliability, RSSM evidence, or third-system evidence.
