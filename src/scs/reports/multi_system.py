from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from scs.reports.claim_audit import compute_claim_audit


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    table = df[columns].copy()
    if table.empty:
        return "| " + " | ".join(columns) + " |\n| " + " | ".join(["---"] * len(columns)) + " |"
    lines = ["| " + " | ".join(columns) + " |"]
    lines.append("| " + " | ".join("---:" if pd.api.types.is_numeric_dtype(table[col]) else "---" for col in columns) + " |")
    for _, row in table.iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.6f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _system_id_from_results(results_dir: Path) -> str:
    risk = pd.read_csv(results_dir / "risk_coverage.csv")
    systems = sorted(set(risk["system_id"].astype(str)))
    if len(systems) != 1:
        raise ValueError(f"{results_dir} must contain exactly one system_id, found {systems}")
    return systems[0]


def _seed_summary_for(system_id: str) -> dict[str, Any] | None:
    candidates = [
        Path(f"results/seed_sweep_{system_id}/seed_sweep_summary.json"),
        Path(f"results/seed_sweep_{system_id.replace('_', '')}/seed_sweep_summary.json"),
    ]
    for candidate in candidates:
        loaded = _load_json_if_exists(candidate)
        if loaded is not None:
            return loaded
    return None


def _severity_summary_for(system_id: str) -> dict[str, Any] | None:
    candidates = [
        Path(f"results/{system_id}_severity_sweep/severity_summary.json"),
        Path(f"results/{system_id}_severity/severity_summary.json"),
    ]
    for candidate in candidates:
        loaded = _load_json_if_exists(candidate)
        if loaded is not None:
            return loaded
    return None


def _overall_status(system_rows: list[dict[str, Any]], gate: dict[str, Any] | None) -> tuple[str, str]:
    if gate is not None and gate.get("decision") == "KILL_OR_DOWNGRADE_CLAIM":
        return (
            "NOT_SUPPORTED",
            "The v0 decision gate is KILL_OR_DOWNGRADE_CLAIM, so expansion outputs are reported as diagnostics only.",
        )
    verdicts = [row["claim_verdict"] for row in system_rows]
    if verdicts and all(verdict == "SUPPORTED" for verdict in verdicts):
        return "SUPPORTED", "Every evaluated system has a SUPPORTED claim audit verdict."
    if any(verdict in {"SUPPORTED", "MIXED"} for verdict in verdicts):
        return "MIXED", "At least one evaluated system has partial or supported evidence, but not all systems do."
    return "NOT_SUPPORTED", "No evaluated system supports the combined-judge claim."


def make_multi_system_report(
    results_dirs: list[str | Path],
    output: str | Path,
    gate_path: str | Path = "reports/v0_decision_gate.json",
) -> dict[str, Any]:
    if len(results_dirs) < 2:
        raise ValueError("multi-system report requires at least two result directories")

    system_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    systems: list[str] = []
    models: set[str] = set()
    judges: set[str] = set()

    for raw_dir in results_dirs:
        results_dir = Path(raw_dir)
        if not (results_dir / "risk_coverage.csv").exists():
            raise FileNotFoundError(results_dir / "risk_coverage.csv")
        system_id = _system_id_from_results(results_dir)
        audit, summary = compute_claim_audit(results_dir)
        risk = pd.read_csv(results_dir / "risk_coverage.csv")
        systems.append(system_id)
        models.update(risk["model_id"].astype(str).unique())
        judges.update(risk["judge_id"].astype(str).unique())
        seed_summary = _seed_summary_for(system_id)
        severity_summary = _severity_summary_for(system_id)
        system_rows.append(
            {
                "system": system_id,
                "claim_verdict": summary["verdict"],
                "seed_verdict": seed_summary["verdict"] if seed_summary is not None else "NOT_RUN",
                "severity_verdict": severity_summary["verdict"] if severity_summary is not None else "NOT_RUN",
            }
        )
        comparison_rows.append(
            {
                "system": system_id,
                "combined_win_rate": float(summary["overall_win_rate"]),
                "best_simple_judge": summary["best_simple_judge_overall"],
            }
        )
        audit.to_csv(results_dir / "claim_audit.csv", index=False)
        (results_dir / "claim_audit.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    gate = _load_json_if_exists(Path(gate_path))
    overall, explanation = _overall_status(system_rows, gate)
    known_failures = []
    if gate is not None and gate.get("decision") != "PROCEED_TO_CSTR":
        known_failures.append(f"Decision gate is {gate.get('decision')}; expansion is diagnostic, not claim support.")
    if any(row["claim_verdict"] == "NOT_SUPPORTED" for row in system_rows):
        known_failures.append("At least one system has claim_verdict=NOT_SUPPORTED.")

    report = {
        "systems": sorted(systems),
        "models": sorted(models),
        "judges": sorted(judges),
        "claim_status_by_system": system_rows,
        "combined_vs_best_simple": comparison_rows,
        "overall_claim_status": overall,
        "explanation": explanation,
        "decision_gate": gate.get("decision") if gate is not None else "MISSING",
        "known_failures": known_failures,
    }

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    system_df = pd.DataFrame(system_rows)
    comparison_df = pd.DataFrame(comparison_rows)
    text = f"""# Multi-System Report

## Systems

{", ".join(report["systems"])}

## Models

{", ".join(report["models"])}

## Judges

{", ".join(report["judges"])}

## Claim status by system

{_markdown_table(system_df, ["system", "claim_verdict", "seed_verdict", "severity_verdict"])}

## Combined judge vs strongest simple judge

{_markdown_table(comparison_df, ["system", "combined_win_rate", "best_simple_judge"])}

## Overall claim status

{overall}

## Explanation

{explanation}

## Known failures

{"- none" if not known_failures else chr(10).join(f"- {item}" for item in known_failures)}
"""
    output_path.write_text(text, encoding="utf-8")
    return report
