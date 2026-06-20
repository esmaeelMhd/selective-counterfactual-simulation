# v0 Freeze Report

## Commit

02b6aa2

## Environment

- Python: 3.13.11
- Platform: Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39

## Commands run

- git status
- python -m venv .venv
- source .venv/bin/activate
- pip install -e ".[dev]"
- pytest -q
- python scripts/run_smoke.py

## Pytest result

passed

## Smoke result

passed

## Created artifacts

- results/smoke_two_tank/data_summary.json: exists=True size=1861
- results/smoke_two_tank/model_metrics.csv: exists=True size=1760
- results/smoke_two_tank/scenario_scores.csv: exists=True size=226745
- results/smoke_two_tank/risk_coverage.csv: exists=True size=62137
- results/smoke_two_tank/risk_coverage.png: exists=True size=97074
- results/smoke_two_tank/summary.json: exists=True size=9032
- reports/smoke_report.md: exists=True size=5889

## Forbidden dependency scan

- forbidden repo references: 0
- path hacks: 0

## Known failures

- none

## Verdict

ACCEPTED
