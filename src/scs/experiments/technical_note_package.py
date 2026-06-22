from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from scs.experiments.current_status import (
    README_END,
    README_START,
    _git_dirty_lines,
    _markdown_table,
    _scan_forbidden_runtime_refs,
)


PACKAGE_RESULTS_ROOT = Path("results/technical_note_package")
TECHNICAL_NOTE_TITLE = "When Should a Learned Simulator Refuse? A Weak-Positive Benchmark on Synthetic Dynamical Systems"
EXACT_CLAIM_SENTENCE = "The current evidence supports only a weak but positive low-coverage refusal result under a frozen protocol."
REQUIRED_FIGURES = [
    "main_low_coverage_margins.png",
    "twotank_vs_cstr_far.png",
    "signal_role_summary.png",
    "smoke_model_sanity.png",
]
PROTECTED_SOURCE_ARTIFACTS = [
    "results/calibrated_two_tank/",
    "results/calibrated_cstr/",
    "results/effect_size_audit/",
    "results/cstr_weakness_audit/",
    "results/repair_signal_semantics_audit/",
    "reports/practical_utility_decision_gate.md",
    "reports/repair_signal_role_decision_gate.md",
    "reports/current_status_decision_gate.md",
]
BUILT_IN_FORBIDDEN_PHRASES = [
    "strong support",
    "general selective simulation",
    "trustworthy simulator",
    "trustworthy counterfactual simulation",
    "safety certification",
    "product readiness",
    "product-ready",
    "validated digital twin",
    "industrial ai breakthrough",
    "high-coverage reliability",
    "plant-wide simulation",
    "plant-wide digital twin",
    "autonomous control",
    "universal simulator",
    "broad simulator reliability",
    "general simulator reliability",
]
ALLOWED_CLAIM_CONTEXTS = [
    "non-claims",
    "limitations",
    "forbidden claims",
    "what it does not claim",
    "what is not supported",
    "what this benchmark does not test",
    "important negative findings",
    "negative findings",
    "negative result",
    "weakness",
    "signal semantics",
    "honest limitations",
    "what not to do next",
    "claim audit",
    "claim boundaries",
    "non-intended use",
]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    _ensure_dir(target.parent)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_yaml(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return data


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _gate_json(markdown_path: str | Path) -> dict[str, Any]:
    md = Path(markdown_path)
    json_path = md.with_suffix(".json")
    if json_path.exists():
        return _read_json(json_path)
    text = md.read_text(encoding="utf-8")
    return {"text": text}


def _coverage_row(df: pd.DataFrame, coverage: float) -> pd.Series:
    rows = df[np.isclose(df["coverage"].astype(float), coverage)]
    if rows.empty:
        raise ValueError(f"missing coverage {coverage}")
    return rows.iloc[0]


def _word_count(path: str | Path) -> int:
    return len(re.findall(r"\b\w+\b", Path(path).read_text(encoding="utf-8")))


def load_package_config(path: str | Path) -> dict[str, Any]:
    config = _read_yaml(path)
    required = {
        "package_id",
        "source_commit",
        "source_tag",
        "controlling_status",
        "source_artifacts",
        "allowed_claim",
        "required_caveats",
        "forbidden_claims",
        "outputs",
        "forbidden",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"missing technical note package config keys: {missing}")
    if config["package_id"] != "limitations_first_technical_note_v1":
        raise ValueError("unexpected package_id")
    if config["allowed_claim"]["label"] != "WEAK_LOW_COVERAGE_BENCHMARK":
        raise ValueError("allowed claim must remain weak and narrow")
    forbidden = config["forbidden"]
    for key in [
        "allow_new_experiments",
        "allow_new_systems",
        "allow_new_models",
        "allow_new_judges",
        "allow_protocol_mutation",
        "allow_prior_artifact_overwrite",
    ]:
        if forbidden.get(key) is not False:
            raise ValueError(f"forbidden.{key} must be false")
    caveat_text = "\n".join(config["required_caveats"]).lower()
    if "positive but weak" not in caveat_text or "diagnostic-only for cstr" not in caveat_text:
        raise ValueError("required caveats must include CSTR weakness and repair diagnostic-only status")
    return config


def verify_technical_note_preconditions(
    config_path: str | Path,
    output: str | Path,
    report_output: str | Path = "reports/technical_note_precondition_check.md",
) -> dict[str, Any]:
    config = load_package_config(config_path)
    out_dir = Path(output)
    _ensure_dir(out_dir)
    current_gate = _gate_json(config["controlling_status"]["current_status_gate"])
    practical_gate = _gate_json(config["controlling_status"]["practical_utility_gate"])
    repair_gate = _gate_json(config["controlling_status"]["repair_signal_role_gate"])
    source_artifacts = config["source_artifacts"]
    missing = [path for path in source_artifacts.values() if not Path(path).exists() or Path(path).stat().st_size == 0]
    readme_text = Path("README.md").read_text(encoding="utf-8")
    dependency_scan = _scan_forbidden_runtime_refs([Path("src"), Path("scripts")])
    artifact_refs = yaml.safe_dump(source_artifacts).lower()
    forbidden_evidence_refs = []
    if "heat_exchanger" in artifact_refs:
        forbidden_evidence_refs.append("heat_exchanger referenced as evidence")
    if "rssm" in artifact_refs:
        forbidden_evidence_refs.append("RSSM referenced as evidence")
    hashes = {
        key: {"path": path, "sha256": _sha256(path), "bytes": Path(path).stat().st_size}
        for key, path in source_artifacts.items()
        if Path(path).exists()
    }
    _write_json(out_dir / "source_artifact_hashes.json", {"artifacts": hashes})
    expansion_allowed = bool(
        current_gate.get("expansion_allowed")
        or practical_gate.get("expansion_allowed")
        or repair_gate.get("expansion_allowed")
        or any(config["forbidden"].values())
    )
    allowed_next_action = current_gate.get("allowed_next_action") or repair_gate.get("allowed_next_action")
    reasons: list[str] = []
    if missing:
        reasons.append(f"missing source artifacts: {missing}")
    if current_gate.get("decision") != "CURRENT_STATUS_SYNCED":
        reasons.append("current status gate is not CURRENT_STATUS_SYNCED")
    if practical_gate.get("decision") != "NARROW_TO_WEAK_LOW_COVERAGE_CLAIM":
        reasons.append("practical utility gate is not NARROW_TO_WEAK_LOW_COVERAGE_CLAIM")
    if repair_gate.get("decision") != "MARK_REPAIR_DIAGNOSTIC_ONLY_FOR_CSTR":
        reasons.append("repair signal role gate is not MARK_REPAIR_DIAGNOSTIC_ONLY_FOR_CSTR")
    if allowed_next_action not in {"MAINTAIN_REPO_AS_WEAK_POSITIVE_BENCHMARK", "PREPARE_LIMITATIONS_FIRST_TECHNICAL_NOTE", "UPDATE_SIGNAL_SEMANTICS_ONLY"}:
        reasons.append(f"unexpected allowed next action: {allowed_next_action}")
    if expansion_allowed:
        reasons.append("expansion is allowed")
    if README_START not in readme_text or README_END not in readme_text:
        reasons.append("README current status block is missing")
    if dependency_scan["old_repo_runtime_import_hits"] or dependency_scan["path_hack_hits"]:
        reasons.append("forbidden runtime dependency/path scan failed")
    reasons.extend(forbidden_evidence_refs)
    verdict = "READY_FOR_TECHNICAL_NOTE_PACKAGE" if not reasons else "NOT_READY"
    result = {
        "package_id": config["package_id"],
        "working_tree_dirty": bool(_git_dirty_lines()),
        "dirty_state": _git_dirty_lines(),
        "current_status_gate": current_gate.get("decision"),
        "practical_utility_gate": practical_gate.get("decision"),
        "repair_signal_role_gate": repair_gate.get("decision"),
        "allowed_next_action": allowed_next_action,
        "expansion_allowed": expansion_allowed,
        "required_source_artifacts": [{"name": key, "path": path, "exists": path not in missing} for key, path in source_artifacts.items()],
        "source_artifact_hash_manifest": str(out_dir / "source_artifact_hashes.json"),
        "forbidden_dependency_scan": dependency_scan,
        "forbidden_evidence_refs": forbidden_evidence_refs,
        "verdict": verdict,
        "reasons": reasons,
    }
    _write_json(out_dir / "precondition_check.json", result)
    write_technical_note_precondition_report(result, report_output)
    if verdict != "READY_FOR_TECHNICAL_NOTE_PACKAGE":
        raise RuntimeError(f"technical note preconditions failed: {reasons}")
    return result


def write_technical_note_precondition_report(result: dict[str, Any], output: str | Path) -> None:
    artifacts = pd.DataFrame(result["required_source_artifacts"])
    scan = result["forbidden_dependency_scan"]
    text = f"""# Technical Note Package Preconditions

## Working tree

Dirty: {result["working_tree_dirty"]}

## Controlling gates

Current status gate: {result["current_status_gate"]}

Practical utility gate: {result["practical_utility_gate"]}

Repair signal role gate: {result["repair_signal_role_gate"]}

Allowed next action: {result["allowed_next_action"]}

## Expansion status

Expansion allowed: {result["expansion_allowed"]}

## Required source artifacts

{_markdown_table(artifacts, ["name", "path", "exists"])}

## Source artifact hash manifest

{result["source_artifact_hash_manifest"]}

## Forbidden dependency scan

Old repo runtime import hits: {scan["old_repo_runtime_import_hits"] or "none"}

Path hack hits: {scan["path_hack_hits"] or "none"}

Forbidden evidence refs: {result["forbidden_evidence_refs"] or "none"}

## Verdict

{result["verdict"]}
"""
    target = Path(output)
    _ensure_dir(target.parent)
    target.write_text(text, encoding="utf-8")


def extract_evidence_tables(config_path: str | Path) -> dict[str, Any]:
    config = load_package_config(config_path)
    artifacts = config["source_artifacts"]
    current_manifest = _read_json(artifacts["current_manifest"])
    practical_gate = _gate_json(config["controlling_status"]["practical_utility_gate"])
    repair_gate = _gate_json(config["controlling_status"]["repair_signal_role_gate"])
    twotank = pd.read_csv(artifacts["twotank_low_coverage"])
    cstr = pd.read_csv(artifacts["cstr_low_coverage"])
    main_rows: list[dict[str, Any]] = []
    for system_id, frame in [("two_tank", twotank), ("cstr", cstr)]:
        effect_strength = current_manifest["systems"][system_id]["effect_strength"]
        for coverage in [0.05, 0.10]:
            row = _coverage_row(frame, coverage)
            main_rows.append(
                {
                    "system": system_id,
                    "coverage": float(coverage),
                    "baseline_far": float(row["baseline_far"]),
                    "calibrated_far": float(row["calibrated_far"]),
                    "margin": float(row["margin"]),
                    "effect_strength": effect_strength,
                }
            )
    model_metrics = pd.read_csv(artifacts["smoke_model_metrics"])
    id_metrics = model_metrics[model_metrics["split"] == "id_test"].copy()
    model_rows = [
        {"model": row["model_id"], "id_rmse_mean": float(row["rmse_mean"])}
        for _, row in id_metrics.sort_values("model_id").iterrows()
        if row["model_id"] in {"hold_last", "linear_narx", "mlp_state_space"}
    ]
    signal_rows = [
        {
            "signal": "repair_amount",
            "system": "cstr",
            "role": current_manifest["signal_roles"]["repair_amount"]["cstr_role"],
            "key_finding": f"AUROC {current_manifest['signal_roles']['repair_amount']['cstr_repair_auroc']:.6f}; diagnostic-only for within-bound CSTR errors",
        },
        {
            "signal": "invariant_residual",
            "system": "cstr",
            "role": current_manifest["signal_roles"]["invariant_residual"]["cstr_role"],
            "key_finding": f"AUROC {current_manifest['signal_roles']['invariant_residual']['cstr_invariant_auroc']:.6f}; much more informative for CSTR",
        },
    ]
    claim_rows = [
        {
            "claim": "combined_linear works",
            "status": "not supported as the original broad claim",
            "evidence": "v0 decision gate killed or downgraded the original combined_linear claim",
            "allowed_wording": "combined_linear is an exploratory baseline",
        },
        {
            "claim": "calibrated low-coverage works on TwoTank",
            "status": "supported",
            "evidence": "TwoTank low-coverage margins are practically meaningful",
            "allowed_wording": "TwoTank shows a practically meaningful low-coverage result",
        },
        {
            "claim": "calibrated low-coverage weakly replicates on CSTR",
            "status": "weak positive",
            "evidence": "CSTR margins are positive but small",
            "allowed_wording": "CSTR weakly replicates the low-coverage direction",
        },
        {
            "claim": "repair_amount is universal",
            "status": "false",
            "evidence": "CSTR repair AUROC is 0.5 and role gate marks repair diagnostic-only",
            "allowed_wording": "repair_amount is diagnostic-only for CSTR",
        },
        {
            "claim": "invariant_residual is informative on CSTR",
            "status": "supported",
            "evidence": "CSTR invariant residual AUROC is high in the repair-vs-invariant audit",
            "allowed_wording": "invariant_residual is informative for CSTR",
        },
        {
            "claim": "general simulator reliability",
            "status": "forbidden",
            "evidence": "Current status gate blocks expansion and general reliability claims",
            "allowed_wording": "No general simulator reliability claim is supported",
        },
        {
            "claim": "product readiness",
            "status": "forbidden",
            "evidence": "The repo has no product layer and the current status forbids product claims",
            "allowed_wording": "This is not product-ready",
        },
    ]
    summary = {
        "package_id": config["package_id"],
        "allowed_claim": config["allowed_claim"]["text"],
        "effect_size_verdict": current_manifest.get("effect_size_verdict") or practical_gate.get("effect_size_verdict"),
        "practical_utility_gate_decision": practical_gate.get("decision"),
        "repair_signal_role_decision": repair_gate.get("decision"),
        "repair_auroc_cstr": current_manifest["signal_roles"]["repair_amount"]["cstr_repair_auroc"],
        "invariant_residual_auroc_cstr": current_manifest["signal_roles"]["invariant_residual"]["cstr_invariant_auroc"],
        "source_artifacts": artifacts,
    }
    return {
        "main": pd.DataFrame(main_rows),
        "models": pd.DataFrame(model_rows),
        "signals": pd.DataFrame(signal_rows),
        "claims": pd.DataFrame(claim_rows),
        "summary": summary,
    }


def build_technical_note_evidence_tables(
    config_path: str | Path,
    output: str | Path,
    report_output: str | Path = "reports/technical_note_evidence_tables.md",
) -> dict[str, Any]:
    tables = extract_evidence_tables(config_path)
    out_dir = Path(output)
    _ensure_dir(out_dir)
    tables["main"].to_csv(out_dir / "main_result_table.csv", index=False)
    tables["models"].to_csv(out_dir / "model_sanity_table.csv", index=False)
    tables["signals"].to_csv(out_dir / "signal_semantics_table.csv", index=False)
    tables["claims"].to_csv(out_dir / "claim_status_table.csv", index=False)
    _write_json(out_dir / "evidence_tables_summary.json", tables["summary"])
    write_technical_note_evidence_report(tables, report_output)
    return tables["summary"]


def write_technical_note_evidence_report(tables: dict[str, Any], output: str | Path) -> None:
    text = f"""# Technical Note Evidence Tables

## Main low-coverage result

{_markdown_table(tables["main"], ["system", "coverage", "baseline_far", "calibrated_far", "margin", "effect_strength"])}

## Smoke model sanity

{_markdown_table(tables["models"], ["model", "id_rmse_mean"])}

## Signal semantics

{_markdown_table(tables["signals"], ["signal", "system", "role", "key_finding"])}

## Claim status

{_markdown_table(tables["claims"], ["claim", "status", "evidence"])}
"""
    target = Path(output)
    _ensure_dir(target.parent)
    target.write_text(text, encoding="utf-8")


def build_technical_note_figures(
    config_path: str | Path,
    tables_dir: str | Path,
    output: str | Path,
    report_output: str | Path = "reports/technical_note_figures.md",
    manifest_output: str | Path | None = None,
) -> dict[str, Any]:
    _ = load_package_config(config_path)
    tables = Path(tables_dir)
    out_dir = Path(output)
    _ensure_dir(out_dir)
    manifest_path = Path(manifest_output) if manifest_output is not None else PACKAGE_RESULTS_ROOT / "figures" / "figure_manifest.json"
    _ensure_dir(manifest_path.parent)
    main = pd.read_csv(tables / "main_result_table.csv")
    models = pd.read_csv(tables / "model_sanity_table.csv")
    signals = pd.read_csv(tables / "signal_semantics_table.csv")
    _plot_main_margins(main, out_dir / "main_low_coverage_margins.png")
    _plot_far_pairs(main, out_dir / "twotank_vs_cstr_far.png")
    _plot_signal_roles(signals, out_dir / "signal_role_summary.png")
    _plot_model_sanity(models, out_dir / "smoke_model_sanity.png")
    figures = [str(out_dir / name) for name in REQUIRED_FIGURES]
    summary = {
        "verdict": "FIGURES_BUILT",
        "figures": figures,
        "source_tables": {
            "main_result_table": str(tables / "main_result_table.csv"),
            "model_sanity_table": str(tables / "model_sanity_table.csv"),
            "signal_semantics_table": str(tables / "signal_semantics_table.csv"),
        },
    }
    _write_json(manifest_path, summary)
    write_technical_note_figures_report(summary, report_output)
    return summary


def _plot_main_margins(main: pd.DataFrame, output: Path) -> None:
    labels = [f"{row.system}\n{row.coverage:.2f}" for row in main.itertuples()]
    colors = ["#2f6f8f" if row.system == "two_tank" else "#b95f3b" for row in main.itertuples()]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, main["margin"], color=colors)
    ax.set_title("Low-coverage calibrated FAR margin")
    ax.set_ylabel("Baseline FAR - calibrated FAR")
    ax.set_xlabel("System and coverage")
    ax.axhline(0.0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _plot_far_pairs(main: pd.DataFrame, output: Path) -> None:
    x = np.arange(len(main))
    labels = [f"{row.system}\n{row.coverage:.2f}" for row in main.itertuples()]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.18, main["baseline_far"], width=0.36, label="Baseline FAR", color="#677a8a")
    ax.bar(x + 0.18, main["calibrated_far"], width=0.36, label="Calibrated FAR", color="#3b7f5f")
    ax.set_title("Baseline vs calibrated false accept rate")
    ax.set_ylabel("False accept rate")
    ax.set_xlabel("System and coverage")
    ax.set_xticks(x, labels)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _plot_signal_roles(signals: pd.DataFrame, output: Path) -> None:
    labels = [f"{row.signal}\n{row.system}" for row in signals.itertuples()]
    score = [0.35 if row.role == "diagnostic_only" else 0.95 for row in signals.itertuples()]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(labels, score, color=["#b95f3b", "#2f6f8f"])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Current CSTR role strength")
    ax.set_title("Signal semantics under CSTR weakness audit")
    for idx, row in enumerate(signals.itertuples()):
        ax.text(idx, score[idx] + 0.03, row.role, ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _plot_model_sanity(models: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(models["model"], models["id_rmse_mean"], color="#4f6d7a")
    ax.set_title("TwoTank smoke ID model sanity")
    ax.set_ylabel("ID RMSE mean")
    ax.set_xlabel("Model")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def write_technical_note_figures_report(summary: dict[str, Any], output: str | Path) -> None:
    rows = pd.DataFrame([{"figure": figure} for figure in summary["figures"]])
    sources = pd.DataFrame([{"source_table": key, "path": value} for key, value in summary["source_tables"].items()])
    text = f"""# Technical Note Figures

## Figures

{_markdown_table(rows, ["figure"])}

## Source tables

{_markdown_table(sources, ["source_table", "path"])}

## Verdict

{summary["verdict"]}
"""
    target = Path(output)
    _ensure_dir(target.parent)
    target.write_text(text, encoding="utf-8")


def _load_table_bundle(tables_dir: str | Path) -> dict[str, Any]:
    tables = Path(tables_dir)
    return {
        "main": pd.read_csv(tables / "main_result_table.csv"),
        "models": pd.read_csv(tables / "model_sanity_table.csv"),
        "signals": pd.read_csv(tables / "signal_semantics_table.csv"),
        "claims": pd.read_csv(tables / "claim_status_table.csv"),
        "summary": _read_json(tables / "evidence_tables_summary.json"),
    }


def build_limitations_first_technical_note(
    config_path: str | Path,
    tables_dir: str | Path,
    figures_dir: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    config = load_package_config(config_path)
    bundle = _load_table_bundle(tables_dir)
    main = bundle["main"]
    summary = bundle["summary"]
    twotank_005 = _coverage_row(main[main["system"] == "two_tank"], 0.05)
    twotank_010 = _coverage_row(main[main["system"] == "two_tank"], 0.10)
    cstr_005 = _coverage_row(main[main["system"] == "cstr"], 0.05)
    cstr_010 = _coverage_row(main[main["system"] == "cstr"], 0.10)
    fig_dir = Path(figures_dir)
    text = f"""# {TECHNICAL_NOTE_TITLE}

## Abstract

This note packages the current state of the Selective Counterfactual Simulation benchmark. It is limitations-first: the benchmark shows a useful TwoTank low-coverage result, a weaker CSTR replication, and a clear signal-semantics failure for repair amount on CSTR. {EXACT_CLAIM_SENTENCE}

## One-sentence claim

{EXACT_CLAIM_SENTENCE}

## Non-claims

This is not safety certification.
This is not a product-ready digital twin.
This is not a claim of general simulator reliability.
This is not high-coverage reliability.
This is not evidence for RSSM or a third system.

## Motivation

The benchmark asks whether a learned or hybrid simulator can rank counterfactual intervention scenarios by answerability and refuse the riskiest cases. The package is useful as a research-engineering artifact because the conclusion is explicit about where the evidence works and where it does not.

## Benchmark setup

The protocol is frozen around existing synthetic dynamical-system artifacts. The package reads existing result tables and decision gates; it does not introduce new systems, models, judges, or success rules.

## Systems

Two systems provide the current evidence. TwoTank is the stronger result and CSTR is the weakness check. Heat-exchanger results and RSSM results are not part of the evidence package.

## Models

The smoke sanity table includes `hold_last`, `linear_narx`, and `mlp_state_space`. On TwoTank ID smoke data, the ID RMSE means are:

{_markdown_table(bundle["models"], ["model", "id_rmse_mean"])}

## Refusal signals

The calibrated judge uses existing refusal signals. The important semantic update is that repair amount is not universal.

{_markdown_table(bundle["signals"], ["signal", "system", "role", "key_finding"])}

## Calibration protocol

The calibrated protocol selects low-risk scenarios under frozen train/calibration/test separation and evaluates false accept rate at fixed coverage. The practical utility gate decision is `{summary["practical_utility_gate_decision"]}`.

## Primary metric

The primary metric is false accept rate at fixed coverage. Lower false accept rate at the same coverage is better.

## Main result

![Low coverage margins](figures/main_low_coverage_margins.png)

TwoTank has a coverage 0.05 margin of {twotank_005["margin"]:.6f} and a coverage 0.10 margin of {twotank_010["margin"]:.6f}. CSTR has a coverage 0.05 margin of {cstr_005["margin"]:.6f} and a coverage 0.10 margin of {cstr_010["margin"]:.6f}. The effect-size verdict is `{summary["effect_size_verdict"]}`.

![Baseline vs calibrated FAR](figures/twotank_vs_cstr_far.png)

## Negative result: combined_linear failed

The original v0 `combined_linear` claim was downgraded. The current package treats it as an exploratory baseline, not as a supported broad refusal method.

## Weakness: CSTR effect is positive but small

CSTR is positive at low coverage, but the margins are small: {cstr_005["margin"]:.6f} at coverage 0.05 and {cstr_010["margin"]:.6f} at coverage 0.10. This is why the current allowed claim remains weak.

## Signal semantics: repair is not universal

The repair role decision is `{summary["repair_signal_role_decision"]}`. CSTR repair AUROC is {summary["repair_auroc_cstr"]:.6f}, while CSTR invariant residual AUROC is {summary["invariant_residual_auroc_cstr"]:.6f}. repair_amount is diagnostic-only for CSTR; invariant_residual is much more informative for CSTR.

![Signal role summary](figures/signal_role_summary.png)

## Failure analysis summary

The CSTR failure mode is mostly within-bound dynamic error, so a bounds/projection signal does not separate the bad accepted cases. The invariant residual is closer to the actual failure mechanism.

## Limitations

- The result is low-coverage only.
- CSTR is positive but weak.
- Expansion is blocked.
- No safety, product, or general reliability claim is supported.
- No RSSM, third-system, high-coverage, or plant-wide claim is supported.

## Reproducibility

Run:

```bash
pip install -e ".[dev]"
pytest -q
python scripts/run_smoke.py
python scripts/check_technical_note_package.py --config configs/status/technical_note_package.yaml --manifest results/technical_note_package/package_manifest.json
```

Main source figures are in `{fig_dir}`.

![Smoke model sanity](figures/smoke_model_sanity.png)

## Conclusion

This package should be read as a disciplined weak-positive benchmark state, not as a broad simulator-reliability result. The most useful next action is to maintain the repo as a weak-positive benchmark and use the negative findings to guide future paper planning.
"""
    target = Path(output)
    _ensure_dir(target.parent)
    target.write_text(text, encoding="utf-8")
    return {"path": str(target), "title": TECHNICAL_NOTE_TITLE, "allowed_claim": config["allowed_claim"]["text"]}


def build_portfolio_package(
    config_path: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    report_output: str | Path = "reports/portfolio_package_report.md",
) -> dict[str, Any]:
    config = load_package_config(config_path)
    manifest = _read_json(manifest_path)
    out_dir = Path(output_dir)
    _ensure_dir(out_dir)
    one_page = out_dir / "one_page_project_summary.md"
    portfolio = out_dir / "portfolio_summary.md"
    claim_table = out_dir / "claim_audit_table.md"
    repro = out_dir / "reproducibility_card.md"
    allowed_claim = config["allowed_claim"]["text"]
    one_page.write_text(
        f"""# Selective Counterfactual Simulation Benchmark

## What it is

A Python research benchmark for testing when learned simulators should answer or refuse counterfactual intervention scenarios.

## What problem it tests

The benchmark tests selective prediction under intervention shift: accept low-risk scenarios and abstain on scenarios likely to produce materially wrong rollouts.

## What I built

I built synthetic TwoTank and CSTR systems, data generation, three simulator models, refusal signals, calibrated judge selection, risk-coverage metrics, evidence audits, and reproducible reports.

## Key result

{allowed_claim} TwoTank is practically meaningful; CSTR is positive but weak.

## Important negative findings

The original `combined_linear` claim was downgraded. `repair_amount` is diagnostic-only for CSTR because it misses within-bound dynamic errors. `invariant_residual` is much more informative for CSTR.

## What it does not claim

This is not safety certification, product readiness, autonomous control, plant-wide simulation, high-coverage reliability, RSSM evidence, or third-system evidence.

## Reproducibility

Run `pip install -e ".[dev]"`, `pytest -q`, and `python scripts/run_smoke.py`. The current status gate is `{manifest["status_id"]}` with expansion blocked.
""",
        encoding="utf-8",
    )
    portfolio.write_text(
        """# Portfolio Summary

## Project headline

Built a limitations-first selective simulation benchmark for counterfactual intervention shift.

## Technical stack

Python, NumPy, pandas, scikit-learn, matplotlib, pytest, YAML configs, and scriptable evidence pipelines.

## Research skills demonstrated

Benchmark framing, metric definition, protocol freezing, calibration/test separation, negative-result analysis, and claim downgrade discipline.

## Engineering skills demonstrated

End-to-end package structure, reproducible CLI scripts, typed simulator/model interfaces, automated reports, plots, and regression tests.

## Evidence discipline demonstrated

The project records decision gates, source artifact hashes, claim-language guards, and explicit non-claims. It keeps weak evidence weak instead of renaming it as success.

## Honest limitations

The current result is low-coverage only. CSTR is positive but weak. repair_amount is diagnostic-only for CSTR. Expansion is blocked until stronger evidence exists.
""",
        encoding="utf-8",
    )
    claim_rows = extract_evidence_tables(config_path)["claims"]
    claim_table.write_text(
        "# Claim Audit Table\n\n"
        + _markdown_table(claim_rows, ["claim", "status", "evidence", "allowed_wording"])
        + "\n",
        encoding="utf-8",
    )
    repro.write_text(
        f"""# Reproducibility Card

## Repo state

Commit: {config["source_commit"]}

Tag: {config["source_tag"]}

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
""",
        encoding="utf-8",
    )
    summary = {
        "verdict": "PORTFOLIO_PACKAGE_BUILT",
        "outputs": [str(one_page), str(portfolio), str(claim_table), str(repro)],
        "one_page_word_count": _word_count(one_page),
    }
    write_portfolio_package_report(summary, report_output)
    return summary


def write_portfolio_package_report(summary: dict[str, Any], output: str | Path) -> None:
    rows = pd.DataFrame([{"path": path} for path in summary["outputs"]])
    text = f"""# Portfolio Package Report

## Outputs

{_markdown_table(rows, ["path"])}

## One-page word count

{summary["one_page_word_count"]}

## Verdict

{summary["verdict"]}
"""
    target = Path(output)
    _ensure_dir(target.parent)
    target.write_text(text, encoding="utf-8")


def build_release_package_manifest(
    config_path: str | Path,
    output: str | Path,
    release_note_output: str | Path = "reports/release_note_v1_current_status.md",
) -> dict[str, Any]:
    config = load_package_config(config_path)
    source_hashes = {
        key: {"path": path, "sha256": _sha256(path), "bytes": Path(path).stat().st_size}
        for key, path in config["source_artifacts"].items()
    }
    included_docs = [
        config["outputs"]["technical_note"],
        config["outputs"]["one_page_summary"],
        config["outputs"]["portfolio_summary"],
        config["outputs"]["claim_audit_table"],
        config["outputs"]["reproducibility_card"],
    ]
    included_reports = [
        "reports/technical_note_precondition_check.md",
        "reports/technical_note_evidence_tables.md",
        "reports/technical_note_figures.md",
        "reports/portfolio_package_report.md",
        str(release_note_output),
    ]
    figure_dir = Path(config["outputs"]["figure_dir"])
    included_figures = [str(figure_dir / name) for name in REQUIRED_FIGURES]
    payload = {
        "release_name": "v1-current-status-sync",
        "release_type": "weak_positive_benchmark",
        "allowed_claim": config["allowed_claim"]["text"],
        "forbidden_claims": config["forbidden_claims"],
        "included_docs": included_docs,
        "included_reports": included_reports,
        "included_figures": included_figures,
        "source_artifact_hashes": source_hashes,
        "reproducibility_commands": [
            'pip install -e ".[dev]"',
            "pytest -q",
            "python scripts/run_smoke.py",
            "python scripts/check_technical_note_package.py --config configs/status/technical_note_package.yaml --manifest results/technical_note_package/package_manifest.json",
        ],
        "known_limitations": config["required_caveats"],
    }
    _write_json(output, payload)
    write_release_note(payload, release_note_output)
    return payload


def write_release_note(payload: dict[str, Any], output: str | Path) -> None:
    text = f"""# Release Note: Weak-Positive Low-Coverage Refusal Benchmark

## What changed

This release packages the current benchmark status into a limitations-first technical note, one-page summary, portfolio summary, claim audit table, reproducibility card, and release manifest.

## Current allowed claim

{payload["allowed_claim"]}

## Main evidence

TwoTank is practically meaningful at low coverage. CSTR is positive but weak. The package uses the frozen current-status evidence and does not add new experiments.

## Negative findings

The original `combined_linear` claim was downgraded. repair_amount is diagnostic-only for CSTR. invariant_residual is the more informative CSTR signal.

## Limitations

{chr(10).join(f"- {item}" for item in payload["known_limitations"])}

## Reproducibility

{chr(10).join(f"- `{command}`" for command in payload["reproducibility_commands"])}

## What not to do next

Expansion is blocked. Maintain the repository as a weak-positive benchmark; do not claim safety certification, product readiness, high-coverage reliability, RSSM evidence, or third-system evidence.
"""
    target = Path(output)
    _ensure_dir(target.parent)
    target.write_text(text, encoding="utf-8")


def scan_forbidden_claim_language(paths: list[str | Path], forbidden_claims: list[str]) -> dict[str, Any]:
    phrases = sorted({*(phrase.lower() for phrase in forbidden_claims), *BUILT_IN_FORBIDDEN_PHRASES})
    violations: list[dict[str, Any]] = []
    allowed: list[dict[str, Any]] = []
    scanned: list[str] = []
    for root in [Path(path) for path in paths]:
        candidates = [root] if root.is_file() else sorted(path for path in root.rglob("*") if path.is_file())
        for path in candidates:
            if path.suffix.lower() not in {".md", ".json", ".yaml", ".yml", ".txt"}:
                continue
            if path == Path("reports/claim_language_scan.md"):
                continue
            scanned.append(str(path))
            if path.suffix.lower() == ".json":
                try:
                    _scan_json_claim_value(path, json.loads(path.read_text(encoding="utf-8")), "", phrases, violations, allowed)
                    continue
                except json.JSONDecodeError:
                    pass
            _scan_markdown_claim_file(path, phrases, violations, allowed)
    return {"paths_scanned": scanned, "violations": violations, "allowed_mentions": allowed, "risk_phrases": phrases}


def _scan_markdown_claim_file(
    path: Path,
    phrases: list[str],
    violations: list[dict[str, Any]],
    allowed: list[dict[str, Any]],
) -> None:
    current_heading = ""
    for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            current_heading = stripped.lstrip("#").strip().lower()
        elif stripped.startswith("**") and "**" in stripped[2:]:
            current_heading = stripped.split("**", 2)[1].strip(": ").lower()
        low = line.lower()
        for phrase in phrases:
            if phrase not in low:
                continue
            record = {"path": str(path), "line": lineno, "phrase": phrase, "context_heading": current_heading}
            if _claim_context_allowed(low, current_heading):
                allowed.append(record)
            else:
                violations.append(record)


def _scan_json_claim_value(
    path: Path,
    value: Any,
    key_path: str,
    phrases: list[str],
    violations: list[dict[str, Any]],
    allowed: list[dict[str, Any]],
) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _scan_json_claim_value(path, child, f"{key_path}.{key}" if key_path else str(key), phrases, violations, allowed)
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _scan_json_claim_value(path, child, f"{key_path}[{index}]", phrases, violations, allowed)
        return
    if not isinstance(value, str):
        return
    low = value.lower()
    heading = key_path.lower().replace("_", " ").replace("-", " ")
    for phrase in phrases:
        if phrase not in low:
            continue
        record = {"path": str(path), "line": None, "phrase": phrase, "context_heading": heading}
        if _claim_context_allowed(low, heading):
            allowed.append(record)
        else:
            violations.append(record)


def _claim_context_allowed(line: str, heading: str) -> bool:
    normalized = heading.replace("_", " ").replace("-", " ")
    allowed_heading = any(token in normalized for token in ALLOWED_CLAIM_CONTEXTS)
    negated = any(token in line for token in ["not ", "no ", "forbidden", "blocked", "does not", "do not", "unsupported", "false", "diagnostic-only"])
    return allowed_heading or negated


def check_technical_note_package(
    config_path: str | Path,
    manifest_path: str | Path,
    output: str | Path = "results/technical_note_package/package_check.json",
    report_output: str | Path = "reports/technical_note_package_check.md",
) -> dict[str, Any]:
    config = load_package_config(config_path)
    manifest = _read_json(manifest_path)
    required_paths = [
        config["outputs"]["technical_note"],
        config["outputs"]["one_page_summary"],
        config["outputs"]["portfolio_summary"],
        config["outputs"]["claim_audit_table"],
        config["outputs"]["reproducibility_card"],
        "reports/release_note_v1_current_status.md",
        "reports/portfolio_package_report.md",
        "reports/technical_note_evidence_tables.md",
        "reports/technical_note_figures.md",
        *manifest.get("included_docs", []),
        *manifest.get("included_reports", []),
        *manifest.get("included_figures", []),
    ]
    required_paths.extend([str(Path(config["outputs"]["figure_dir"]) / name) for name in REQUIRED_FIGURES])
    missing = sorted({path for path in required_paths if not Path(path).exists() or Path(path).stat().st_size == 0})
    readme_text = Path("README.md").read_text(encoding="utf-8")
    readme_synced = README_START in readme_text and README_END in readme_text and "A weak but positive low-coverage result under the frozen protocol." in readme_text
    claim_scan = scan_forbidden_claim_language(
        [
            config["outputs"]["technical_note"],
            config["outputs"]["one_page_summary"],
            config["outputs"]["portfolio_summary"],
            config["outputs"]["claim_audit_table"],
            config["outputs"]["reproducibility_card"],
            "reports/release_note_v1_current_status.md",
        ],
        config["forbidden_claims"],
    )
    source_hash_mismatches = []
    precondition_hashes = _read_json(PACKAGE_RESULTS_ROOT / "preconditions" / "source_artifact_hashes.json")["artifacts"]
    for key, item in precondition_hashes.items():
        current = _sha256(item["path"])
        if current != item["sha256"]:
            source_hash_mismatches.append(item["path"])
    table_mismatches = compare_tables_to_sources(config_path, PACKAGE_RESULTS_ROOT / "evidence_tables")
    prior_mutation_detected = bool(source_hash_mismatches)
    reasons = []
    if missing:
        reasons.append(f"missing package files: {missing}")
    if not readme_synced:
        reasons.append("README current status is stale")
    if claim_scan["violations"]:
        reasons.append("forbidden positive claim language detected")
    if source_hash_mismatches:
        reasons.append(f"source artifact hashes changed: {source_hash_mismatches}")
    if table_mismatches:
        reasons.append(f"evidence tables mismatch source artifacts: {table_mismatches}")
    verdict = "TECHNICAL_NOTE_PACKAGE_ACCEPTED" if not reasons else "TECHNICAL_NOTE_PACKAGE_REJECTED"
    result = {
        "verdict": verdict,
        "missing": missing,
        "readme_synced": readme_synced,
        "claim_language": claim_scan,
        "source_hash_mismatches": source_hash_mismatches,
        "table_mismatches": table_mismatches,
        "prior_artifact_mutation_detected": prior_mutation_detected,
        "reasons": reasons,
    }
    _write_json(output, result)
    write_technical_note_package_check_report(result, report_output)
    if verdict != "TECHNICAL_NOTE_PACKAGE_ACCEPTED":
        raise RuntimeError(f"technical note package rejected: {reasons}")
    return result


def compare_tables_to_sources(config_path: str | Path, tables_dir: str | Path) -> list[str]:
    expected = extract_evidence_tables(config_path)
    actual_main = pd.read_csv(Path(tables_dir) / "main_result_table.csv")
    actual_models = pd.read_csv(Path(tables_dir) / "model_sanity_table.csv")
    mismatches = []
    for column in ["baseline_far", "calibrated_far", "margin"]:
        if not np.allclose(expected["main"][column].to_numpy(), actual_main[column].to_numpy()):
            mismatches.append(f"main_result_table.{column}")
    if not np.allclose(expected["models"]["id_rmse_mean"].to_numpy(), actual_models["id_rmse_mean"].to_numpy()):
        mismatches.append("model_sanity_table.id_rmse_mean")
    return mismatches


def write_technical_note_package_check_report(result: dict[str, Any], output: str | Path) -> None:
    violations = pd.DataFrame(result["claim_language"]["violations"]) if result["claim_language"]["violations"] else pd.DataFrame(columns=["path", "line", "phrase", "context_heading"])
    text = f"""# Technical Note Package Check

## Missing files

{result["missing"] or "none"}

## README status

Synced: {result["readme_synced"]}

## Claim language violations

{_markdown_table(violations, ["path", "line", "phrase", "context_heading"])}

## Source artifact hash mismatches

{result["source_hash_mismatches"] or "none"}

## Evidence table mismatches

{result["table_mismatches"] or "none"}

## Prior artifact mutation detected

{result["prior_artifact_mutation_detected"]}

## Verdict

{result["verdict"]}
"""
    target = Path(output)
    _ensure_dir(target.parent)
    target.write_text(text, encoding="utf-8")
