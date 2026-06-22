# Reproducibility Card

## Repo state

Commit: e0dbdc2

Tag: v1-current-status-sync

## Commands

```bash
pip install -e ".[dev]"
pytest -q
python scripts/run_smoke.py
python scripts/verify_current_status_preconditions.py --config configs/status/current_evidence_status.yaml --output results/current_status/preconditions
python scripts/check_technical_note_package.py --config configs/status/technical_note_package.yaml --manifest results/technical_note_package/package_manifest.json
```

## Main artifacts

- docs/technical_note_limitations_first.md
- docs/one_page_project_summary.md
- docs/portfolio_summary.md
- reports/current_status_decision_gate.md
- reports/technical_note_package_check.md

## Known limitations

- Low-coverage only.
- CSTR effect is positive but weak.
- repair_amount is diagnostic-only for CSTR.
- No safety, product, high-coverage, RSSM, or third-system claim.
