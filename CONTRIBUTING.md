# Contributing

## Scope

Contributions should keep the repository focused on benchmark reproducibility, custom model adapters, tests, and clear documentation.

## Claim discipline

Do not add claims of safety certification, product readiness, broad reliability, or general simulator validity. New evidence should describe failures and limitations plainly.

## Development checks

Before opening a change, run:

```bash
pytest -q
python scripts/check_public_release_ready.py --output results/public_release_audit
```

## Public data hygiene

Do not commit credentials, tokens, customer data, private paths, local notebooks with sensitive paths, or generated caches.

## Pull request notes

Describe what changed, which commands passed, and whether the change affects public claims. If a command could not be run, state why.
