from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def make_claim_downgrade(
    decision_path: str | Path = "reports/v0_decision_gate.json",
    freeze_path: str | Path = "reports/v0_freeze_report.json",
    claim_path: str | Path = "results/smoke_two_tank/claim_audit.json",
    seed_path: str | Path = "results/seed_sweep_two_tank/seed_sweep_summary.json",
    severity_path: str | Path = "results/two_tank_severity_sweep/severity_summary.json",
    output_md: str | Path = "reports/v0_claim_downgrade.md",
    output_json: str | Path = "reports/v0_claim_downgrade.json",
) -> dict[str, Any]:
    decision = _load(decision_path)
    freeze = _load(freeze_path)
    claim = _load(claim_path)
    seed = _load(seed_path)
    severity = _load(severity_path)

    gate_decision = decision["decision"]
    if gate_decision != "KILL_OR_DOWNGRADE_CLAIM":
        action = "NO_DOWNGRADE_REQUIRED"
        downgraded_claim = "No downgrade was generated because the decision gate did not require one."
        next_action = decision.get("required_next_action", "")
    else:
        action = "DOWNGRADE_CLAIM"
        downgraded_claim = (
            "The current v0 evidence does not support the claim that combined_linear "
            "robustly reduces false accepts against the strongest simple judge. "
            "combined_linear remains an exploratory baseline, not a supported method."
        )
        next_action = (
            "Fix v0 first: redesign or calibrate the combined judge, rerun claim audit, "
            "seed sweep, severity sweep, and decision gate before treating expansion "
            "results as evidence for the primary claim."
        )

    report = {
        "action": action,
        "previous_claim": (
            "A combined refusal judge reduces false acceptance compared with simple "
            "uncertainty-only, support-only, and disagreement-only judges."
        ),
        "downgraded_claim": downgraded_claim,
        "evidence": {
            "freeze_result": freeze["verdict"],
            "claim_audit_result": claim["verdict"],
            "claim_overall_win_rate": claim.get("overall_win_rate"),
            "seed_robustness_result": seed["verdict"],
            "seed_mean_win_rate": seed.get("aggregate", {}).get("overall_combined_win_rate", {}).get("mean"),
            "severity_sweep_result": severity["verdict"],
            "decision_gate": gate_decision,
        },
        "blocked_actions": [
            "Do not use current v0 as positive support for the primary claim.",
            "Do not proceed to additional systems as claim-support evidence until v0 is fixed.",
            "Do not add RSSM as a distraction from the failed refusal-ranking evidence.",
        ],
        "next_action": next_action,
    }

    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    evidence = report["evidence"]
    text = f"""# v0 Claim Downgrade

## Decision

{action}

## Previous claim

{report["previous_claim"]}

## Downgraded claim

{downgraded_claim}

## Evidence

| check | result |
|---|---|
| freeze result | {evidence["freeze_result"]} |
| claim audit result | {evidence["claim_audit_result"]} |
| claim overall win rate | {evidence["claim_overall_win_rate"]} |
| seed robustness result | {evidence["seed_robustness_result"]} |
| seed mean win rate | {evidence["seed_mean_win_rate"]} |
| severity sweep result | {evidence["severity_sweep_result"]} |
| decision gate | {evidence["decision_gate"]} |

## Blocked actions

{chr(10).join(f"- {item}" for item in report["blocked_actions"])}

## Required next action

{next_action}
"""
    output_md = Path(output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(text, encoding="utf-8")
    return report

