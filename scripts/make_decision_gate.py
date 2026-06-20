from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def make_decision_gate() -> dict:
    freeze = _load("reports/v0_freeze_report.json")
    claim = _load("results/smoke_two_tank/claim_audit.json")
    seed = _load("results/seed_sweep_two_tank/seed_sweep_summary.json")
    severity = _load("results/two_tank_severity_sweep/severity_summary.json")

    freeze_result = freeze["verdict"]
    claim_result = claim["verdict"]
    seed_result = seed["verdict"]
    severity_result = severity["verdict"]

    if (
        freeze_result == "ACCEPTED"
        and claim_result in {"SUPPORTED", "MIXED"}
        and seed_result in {"ROBUST", "UNSTABLE"}
        and severity_result in {"MEANINGFUL", "PARTIAL"}
    ):
        decision = "PROCEED_TO_CSTR"
        next_action = "Run the controlled CSTR expansion and multi-system report."
    elif (
        freeze_result != "ACCEPTED"
        or (claim_result == "NOT_SUPPORTED" and seed_result == "NOT_SUPPORTED")
        or severity_result == "WEAK"
    ):
        decision = "KILL_OR_DOWNGRADE_CLAIM"
        next_action = "Downgrade the main claim or redesign the benchmark before expansion."
    else:
        decision = "FIX_V0_FIRST"
        next_action = "Inspect weak audit dimensions and improve v0 before expansion."

    report = {
        "freeze_result": freeze_result,
        "claim_audit_result": claim_result,
        "seed_robustness_result": seed_result,
        "severity_sweep_result": severity_result,
        "decision": decision,
        "reasoning": (
            f"Freeze={freeze_result}; claim={claim_result}; seed={seed_result}; "
            f"severity={severity_result}. Decision follows the configured gate rules."
        ),
        "required_next_action": next_action,
    }
    Path("reports/v0_decision_gate.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    text = f"""# v0 Decision Gate

## Freeze result

{freeze_result}

## Claim audit result

{claim_result}

## Seed robustness result

{seed_result}

## Severity sweep result

{severity_result}

## Decision

{decision}

## Reasoning

{report["reasoning"]}

## Required next action

{next_action}
"""
    Path("reports/v0_decision_gate.md").write_text(text, encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the v0 decision gate report.")
    parser.parse_args()
    report = make_decision_gate()
    print(report["decision"])


if __name__ == "__main__":
    main()

