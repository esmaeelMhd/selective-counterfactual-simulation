# Reproducibility Card

## Environment

Use Python 3.11 or newer.

## Fresh checkout commands

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
python scripts/run_smoke.py
python scripts/run_current_status_demo.py
python examples/custom_model_example.py --output results/custom_model_example
```

## Release checks

```bash
python scripts/check_public_release_ready.py --output results/public_release_audit
python scripts/check_fresh_clone_repro.py --repo-path . --output results/fresh_clone_check
python scripts/check_public_release_package.py --output results/public_release_package_check
```

## Historical technical-note package check

Older portfolio packaging evidence can still be checked with:

```bash
python scripts/check_technical_note_package.py --config configs/status/technical_note_package.yaml --manifest results/technical_note_package/package_manifest.json
```

## Expected public artifacts

- `results/public_release_audit/public_release_check.json`
- `results/fresh_clone_check/command_status.json`
- `results/public_release_package_check/package_check.json`
- `reports/public_release_readiness_audit.md`
- `reports/fresh_clone_reproduction_check.md`
- `reports/public_release_package_check.md`

## Known limitations

The benchmark uses synthetic systems, narrow low-coverage evidence, limited external validation, and v2 event-risk failure evidence. It is not safety certification and not a product-ready digital twin.
