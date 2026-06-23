# v2 Reproducibility Card

Run:

```bash
python scripts/v2_verify_preconditions.py --config configs/v2/v2_scientific_strengthening.yaml --output results/v2_scientific_strengthening/preconditions
python scripts/v2_run_heat_exchanger_sanity.py --config configs/v2/v2_scientific_strengthening.yaml --output results/v2_scientific_strengthening/heat_exchanger_sanity
python scripts/v2_validate_event_targets.py --config configs/v2/v2_scientific_strengthening.yaml --event-config configs/v2/v2_event_targets.yaml --output results/v2_scientific_strengthening/event_targets
python scripts/v2_run_frozen_protocol.py --config configs/v2/v2_scientific_strengthening.yaml --event-config configs/v2/v2_event_targets.yaml --output results/v2_scientific_strengthening/frozen_protocol
python scripts/v2_statistical_audit.py --config configs/v2/v2_scientific_strengthening.yaml --results results/v2_scientific_strengthening/frozen_protocol --output results/v2_scientific_strengthening/statistical_audit
python scripts/v2_make_scientific_decision_gate.py --protocol docs/v2/v2_scientific_protocol_lock.md --heat-sanity results/v2_scientific_strengthening/heat_exchanger_sanity/event_label_checks.json --frozen-run results/v2_scientific_strengthening/frozen_protocol/v2_run_summary.json --stats results/v2_scientific_strengthening/statistical_audit/v2_statistical_summary.json --output reports/v2_scientific_decision_gate.md
```
