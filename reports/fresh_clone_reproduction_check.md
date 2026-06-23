# Fresh Clone Reproduction Check

## Verdict

FRESH_CLONE_REPRO_PASSED

## Commands

- `python -m venv .venv`: exit 0 (2.218s)
- `.venv/bin/python -m pip install -e .[dev]`: exit 0 (22.282s)
- `.venv/bin/python -m pytest -q`: exit 0 (161.828s)
- `.venv/bin/python scripts/run_smoke.py`: exit 0 (22.181s)
- `.venv/bin/python scripts/run_current_status_demo.py`: exit 0 (3.029s)
- `.venv/bin/python examples/custom_model_example.py --output results/custom_model_example`: exit 0 (1.206s)

## Log

results/fresh_clone_check/command_log.txt
