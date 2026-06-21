from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from scs.validators.signal_semantics import signal_semantics_registry, write_signal_semantics_artifacts


STATUS_RESULTS_ROOT = Path("results/current_status")
README_START = "<!-- SCS_CURRENT_STATUS_START -->"
README_END = "<!-- SCS_CURRENT_STATUS_END -->"
PRIOR_HASH_ARTIFACTS = [
    "reports/practical_utility_decision_gate.md",
    "reports/cstr_weakness_diagnosis.md",
    "reports/repair_signal_role_decision_gate.md",
    "results/effect_size_audit/effect_size/effect_size_summary.json",
    "results/cstr_weakness_audit/repair_signal/repair_signal_metrics.json",
    "results/repair_signal_semantics_audit/repair_validation/repair_validation_summary.json",
    "results/repair_signal_semantics_audit/repair_vs_invariant/repair_vs_invariant_summary.json",
]
RISK_PHRASES = [
    "strong support",
    "robust general",
    "general selective simulation",
    "trustworthy simulator",
    "safe simulator",
    "safety certification",
    "product-ready",
    "plant-wide digital twin",
    "autonomous control",
    "industrial AI breakthrough",
    "validated digital twin",
    "high-coverage reliability",
    "universal simulator",
]
ALLOWED_CONTEXT_HEADINGS = [
    "forbidden claims",
    "not supported",
    "limitations",
    "negative findings",
    "non-claims",
    "non-goals",
    "risk phrases",
    "what is not supported",
    "forbidden next actions",
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


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "| " + " | ".join(columns) + " |\n| " + " | ".join(["---"] * len(columns)) + " |"
    frame = df[columns].copy()
    lines = ["| " + " | ".join(columns) + " |"]
    lines.append("| " + " | ".join("---:" if pd.api.types.is_numeric_dtype(frame[col]) else "---" for col in columns) + " |")
    for _, row in frame.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, (float, int)) and not isinstance(value, bool):
                values.append(f"{float(value):.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def load_current_status_config(path: str | Path) -> dict[str, Any]:
    config = _read_yaml(path)
    required = {
        "status_id",
        "source_commit",
        "source_tag",
        "controlling_reports",
        "source_artifacts",
        "current_allowed_claim",
        "current_forbidden_claims",
        "expansion_allowed",
        "forbidden_directions",
        "signal_role_decisions",
        "required_readme_status",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"missing current evidence status config keys: {missing}")
    if config["status_id"] != "current_evidence_status_v1":
        raise ValueError("unexpected status_id")
    if config["expansion_allowed"] is not False:
        raise ValueError("current evidence status must keep expansion_allowed false")
    if config["current_allowed_claim"]["label"] != "WEAK_LOW_COVERAGE_CLAIM":
        raise ValueError("current allowed claim must be weak/narrow")
    repair = config["signal_role_decisions"]["repair_amount"]
    if repair["cstr_role"] != "diagnostic_only" or repair["universal_refusal_signal"] is not False:
        raise ValueError("repair_amount must be diagnostic-only for CSTR and non-universal")
    if config["signal_role_decisions"]["invariant_residual"]["cstr_role"] != "informative_refusal_signal":
        raise ValueError("invariant_residual CSTR role must be informative_refusal_signal")
    return config


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_dirty_lines() -> list[str]:
    if not Path(".git").exists():
        return []
    result = subprocess.run(["git", "status", "--short"], check=True, capture_output=True, text=True)
    return [line for line in result.stdout.splitlines() if line.strip()]


def _scan_forbidden_runtime_refs(paths: list[Path]) -> dict[str, list[str]]:
    old_repo_names = [
        "time" + "-series" + "-simulator",
        "digital" + "-twin" + "-engine",
        "flux" + "-attention" + "-engine",
        "plant" + "-scenario" + "-compiler",
    ]
    old_repo_hits: list[str] = []
    path_hacks: list[str] = []
    for root in paths:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith(("import ", "from ")) and any(name.replace("-", "_") in stripped or name in stripped for name in old_repo_names):
                    old_repo_hits.append(str(path))
                    break
            for line in text.splitlines():
                stripped = line.strip()
                env_key = "PYTHON" + "PATH"
                if stripped.startswith("sys.path") or stripped.startswith(f"os.environ[\"{env_key}\""):
                    path_hacks.append(str(path))
                    break
    return {"old_repo_runtime_import_hits": sorted(set(old_repo_hits)), "path_hack_hits": sorted(set(path_hacks))}


def verify_current_status_preconditions(config_path: str | Path, output: str | Path) -> dict[str, Any]:
    config = load_current_status_config(config_path)
    out_dir = Path(output)
    _ensure_dir(out_dir)
    required_paths = [
        *config["controlling_reports"].values(),
        *config["source_artifacts"].values(),
        "README.md",
        "src/scs/validators/signal_semantics.py",
        "docs/signal_semantics_registry.md",
    ]
    missing = [path for path in required_paths if not Path(path).exists() or Path(path).stat().st_size == 0]
    practical_json = _read_json("reports/practical_utility_decision_gate.json")
    repair_json = _read_json("reports/repair_signal_role_decision_gate.json")
    scan = _scan_forbidden_runtime_refs([Path("src"), Path("scripts")])
    source_text = yaml.safe_dump({"source_artifacts": config["source_artifacts"], "controlling_reports": config["controlling_reports"]}).lower()
    forbidden_evidence_refs = []
    if "heat_exchanger" in source_text:
        forbidden_evidence_refs.append("heat_exchanger referenced as evidence")
    if "rssm" in source_text:
        forbidden_evidence_refs.append("RSSM referenced as evidence")
    prior_hashes = {
        path: {"sha256": _sha256(path), "bytes": Path(path).stat().st_size}
        for path in PRIOR_HASH_ARTIFACTS
        if Path(path).exists()
    }
    _write_json(out_dir / "prior_artifact_hashes.json", {"artifacts": prior_hashes})
    expansion_allowed = bool(practical_json.get("expansion_allowed", True) or repair_json.get("expansion_allowed", True) or config["expansion_allowed"])
    reasons: list[str] = []
    if missing:
        reasons.append(f"missing artifacts: {missing}")
    if practical_json.get("decision") != "NARROW_TO_WEAK_LOW_COVERAGE_CLAIM":
        reasons.append("practical utility gate decision is not NARROW_TO_WEAK_LOW_COVERAGE_CLAIM")
    if repair_json.get("decision") != "MARK_REPAIR_DIAGNOSTIC_ONLY_FOR_CSTR":
        reasons.append("repair signal role decision is not MARK_REPAIR_DIAGNOSTIC_ONLY_FOR_CSTR")
    if repair_json.get("allowed_next_action") != "UPDATE_SIGNAL_SEMANTICS_ONLY":
        reasons.append("repair signal allowed next action is not UPDATE_SIGNAL_SEMANTICS_ONLY")
    if expansion_allowed:
        reasons.append("expansion is allowed")
    if scan["old_repo_runtime_import_hits"] or scan["path_hack_hits"]:
        reasons.append("forbidden runtime dependency/path scan failed")
    reasons.extend(forbidden_evidence_refs)
    verdict = "READY_FOR_STATUS_SYNC" if not reasons else "NOT_READY"
    result = {
        "status_id": config["status_id"],
        "working_tree_dirty": bool(_git_dirty_lines()),
        "dirty_state": _git_dirty_lines(),
        "practical_utility_decision": practical_json.get("decision"),
        "repair_signal_role_decision": repair_json.get("decision"),
        "allowed_next_action": repair_json.get("allowed_next_action"),
        "expansion_allowed": expansion_allowed,
        "required_artifacts": [{"path": path, "exists": path not in missing} for path in required_paths],
        "prior_artifact_hash_manifest": str(out_dir / "prior_artifact_hashes.json"),
        "forbidden_dependency_scan": scan,
        "forbidden_evidence_refs": forbidden_evidence_refs,
        "verdict": verdict,
        "reasons": reasons,
    }
    _write_json(out_dir / "precondition_check.json", result)
    write_current_status_precondition_report(result, Path("reports/current_status_precondition_check.md"))
    if verdict != "READY_FOR_STATUS_SYNC":
        raise RuntimeError(f"current status preconditions failed: {reasons}")
    return result


def write_current_status_precondition_report(result: dict[str, Any], output: Path) -> None:
    artifacts = pd.DataFrame(result["required_artifacts"])
    scan = result["forbidden_dependency_scan"]
    text = f"""# Current Status Preconditions

## Working tree

Dirty: {result["working_tree_dirty"]}

## Controlling gates

Practical utility gate: {result["practical_utility_decision"]}

Repair signal role gate: {result["repair_signal_role_decision"]}

Allowed next action: {result["allowed_next_action"]}

## Expansion status

Expansion allowed: {result["expansion_allowed"]}

## Required artifacts

{_markdown_table(artifacts, ["path", "exists"])}

## Prior-artifact hash manifest

{result["prior_artifact_hash_manifest"]}

## Forbidden dependency scan

Old repo runtime import hits: {scan["old_repo_runtime_import_hits"] or "none"}

Path hack hits: {scan["path_hack_hits"] or "none"}

Forbidden evidence refs: {result["forbidden_evidence_refs"] or "none"}

## Verdict

{result["verdict"]}
"""
    _ensure_dir(output.parent)
    output.write_text(text, encoding="utf-8")


def build_current_evidence_manifest(config_path: str | Path, output: str | Path) -> dict[str, Any]:
    config = load_current_status_config(config_path)
    out_dir = Path(output)
    _ensure_dir(out_dir)
    practical = _read_json("reports/practical_utility_decision_gate.json")
    cstr_weakness = _read_json("reports/cstr_weakness_diagnosis.json")
    repair_gate = _read_json("reports/repair_signal_role_decision_gate.json")
    repair_validation = _read_json("results/repair_signal_semantics_audit/repair_validation/repair_validation_summary.json")
    repair_vs_invariant = _read_json("results/repair_signal_semantics_audit/repair_vs_invariant/repair_vs_invariant_summary.json")
    signal_ablation = _read_json("results/repair_signal_semantics_audit/signal_set_ablation/signal_set_ablation_summary.json")
    effect = _read_json("results/effect_size_audit/effect_size/effect_size_summary.json")
    systems = {
        "two_tank": _system_manifest("two_tank", "results/calibrated_two_tank/low_coverage_summary.csv", effect),
        "cstr": _system_manifest("cstr", "results/calibrated_cstr/low_coverage_summary.csv", effect),
    }
    manifest = {
        "status_id": config["status_id"],
        "source_commit": config["source_commit"],
        "source_tag": config["source_tag"],
        "controlling_claim_label": config["current_allowed_claim"]["label"],
        "controlling_claim_text": config["current_allowed_claim"]["text"],
        "expansion_allowed": bool(config["expansion_allowed"] or practical.get("expansion_allowed", True) or repair_gate.get("expansion_allowed", True)),
        "allowed_next_action": repair_gate.get("allowed_next_action"),
        "practical_utility_decision": practical.get("decision"),
        "effect_size_verdict": practical.get("effect_size_verdict"),
        "false_accept_forensics_verdict": practical.get("false_accept_forensics_verdict"),
        "event_risk_verdict": practical.get("event_risk_verdict"),
        "cstr_weakness_diagnosis": cstr_weakness.get("final_diagnosis"),
        "repair_validation_verdict": repair_validation.get("verdict"),
        "repair_vs_invariant_verdict": repair_vs_invariant.get("verdict"),
        "signal_set_ablation_verdict": signal_ablation.get("verdict"),
        "repair_signal_role_decision": repair_gate.get("decision"),
        "systems": systems,
        "signal_roles": {
            "repair_amount": {
                "cstr_role": config["signal_role_decisions"]["repair_amount"]["cstr_role"],
                "twotank_role": config["signal_role_decisions"]["repair_amount"]["twotank_role"],
                "reason": config["signal_role_decisions"]["repair_amount"]["cstr_reason"],
                "cstr_repair_auroc": repair_vs_invariant.get("cstr_repair_auroc"),
                "universal_refusal_signal": False,
            },
            "invariant_residual": {
                "cstr_role": config["signal_role_decisions"]["invariant_residual"]["cstr_role"],
                "reason": config["signal_role_decisions"]["invariant_residual"]["cstr_reason"],
                "cstr_invariant_auroc": repair_vs_invariant.get("cstr_invariant_auroc"),
                "universal_refusal_signal": False,
            },
        },
        "forbidden_claims": list(config["current_forbidden_claims"]),
        "forbidden_directions": list(config["forbidden_directions"]),
        "source_artifacts": config["source_artifacts"],
    }
    _write_json(out_dir / "current_evidence_manifest.json", manifest)
    rows = []
    for system_id, item in systems.items():
        rows.append(
            {
                "system_id": system_id,
                "coverage_0_05_margin": item["coverage_0_05_margin"],
                "coverage_0_10_margin": item["coverage_0_10_margin"],
                "effect_strength": item["effect_strength"],
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "current_evidence_manifest.csv", index=False)
    write_current_evidence_manifest_report(manifest, Path("reports/current_evidence_manifest.md"))
    if manifest["expansion_allowed"]:
        raise RuntimeError("current evidence manifest unexpectedly allows expansion")
    return manifest


def _system_manifest(system_id: str, low_path: str | Path, effect: dict[str, Any]) -> dict[str, Any]:
    low = pd.read_csv(low_path)
    effect_rows = [row for row in effect.get("rows", []) if row.get("system_id") == system_id]
    effect_strength = "unknown"
    verdicts = {str(row.get("verdict", "")).upper() for row in effect_rows}
    if "PRACTICALLY_MEANINGFUL" in verdicts:
        effect_strength = "practically_meaningful"
    elif "POSITIVE_BUT_WEAK" in verdicts:
        effect_strength = "positive_but_weak"
    return {
        "coverage_0_05_margin": _coverage_margin(low, 0.05),
        "coverage_0_10_margin": _coverage_margin(low, 0.10),
        "effect_strength": effect_strength,
        "low_coverage_verdicts": sorted(verdicts),
    }


def _coverage_margin(low: pd.DataFrame, coverage: float) -> float:
    row = low[np.isclose(low["coverage"].astype(float), coverage)]
    if row.empty:
        raise ValueError(f"missing low coverage margin for coverage {coverage}")
    return float(row.iloc[0]["margin"])


def write_current_evidence_manifest_report(manifest: dict[str, Any], output: Path) -> None:
    systems = pd.DataFrame(
        [
            {
                "system": system_id,
                "coverage_0_05_margin": values["coverage_0_05_margin"],
                "coverage_0_10_margin": values["coverage_0_10_margin"],
                "effect_strength": values["effect_strength"],
            }
            for system_id, values in manifest["systems"].items()
        ]
    )
    sources = pd.DataFrame([{"artifact": key, "path": value} for key, value in manifest["source_artifacts"].items()])
    text = f"""# Current Evidence Manifest

## Controlling status

{manifest["practical_utility_decision"]}

## Allowed claim

{manifest["controlling_claim_text"]}

## Forbidden claims

{", ".join(manifest["forbidden_claims"])}

## Expansion status

Expansion allowed: {manifest["expansion_allowed"]}

## TwoTank evidence

{_markdown_table(systems[systems["system"] == "two_tank"], ["system", "coverage_0_05_margin", "coverage_0_10_margin", "effect_strength"])}

## CSTR evidence

{_markdown_table(systems[systems["system"] == "cstr"], ["system", "coverage_0_05_margin", "coverage_0_10_margin", "effect_strength"])}

## Repair signal role

{manifest["signal_roles"]["repair_amount"]["cstr_role"]}: {manifest["signal_roles"]["repair_amount"]["reason"]}

## Invariant residual role

{manifest["signal_roles"]["invariant_residual"]["cstr_role"]}: {manifest["signal_roles"]["invariant_residual"]["reason"]}

## Allowed next action

{manifest["allowed_next_action"]}

## Source artifacts

{_markdown_table(sources, ["artifact", "path"])}
"""
    _ensure_dir(output.parent)
    output.write_text(text, encoding="utf-8")


def sync_signal_semantics_status(status_manifest_path: str | Path, output: str | Path) -> dict[str, Any]:
    manifest = _read_json(status_manifest_path)
    artifacts = write_signal_semantics_artifacts("reports/signal_semantics_registry.md")
    registry = signal_semantics_registry()
    repair = registry["repair_amount"]
    invariant = registry["invariant_residual"]
    docs_text = Path("docs/signal_semantics_registry.md").read_text(encoding="utf-8")
    report_text = Path("reports/signal_semantics_registry.md").read_text(encoding="utf-8")
    verdict = "SIGNAL_SEMANTICS_SYNCED"
    reasons = []
    if repair.get("cstr_role") != manifest["signal_roles"]["repair_amount"]["cstr_role"]:
        verdict = "SIGNAL_SEMANTICS_OUT_OF_DATE"
        reasons.append("repair_amount CSTR role mismatch")
    if invariant.get("cstr_role") != manifest["signal_roles"]["invariant_residual"]["cstr_role"]:
        verdict = "SIGNAL_SEMANTICS_OUT_OF_DATE"
        reasons.append("invariant_residual CSTR role mismatch")
    for text_name, text in [("docs", docs_text), ("report", report_text)]:
        if "diagnostic_only" not in text or "informative_refusal_signal" not in text:
            verdict = "SIGNAL_SEMANTICS_OUT_OF_DATE"
            reasons.append(f"{text_name} markdown missing current roles")
    summary = {
        "verdict": verdict,
        "source_manifest": str(status_manifest_path),
        "repair_amount_cstr_role": repair.get("cstr_role"),
        "invariant_residual_cstr_role": invariant.get("cstr_role"),
        "artifacts": artifacts,
        "historical_artifact_policy": "historical calibrated/effect/weakness outputs were not rewritten",
        "reasons": reasons,
    }
    out_dir = Path(output)
    _ensure_dir(out_dir)
    _write_json(out_dir / "signal_semantics_sync_summary.json", summary)
    write_signal_semantics_sync_report(summary, Path("reports/signal_semantics_status_sync.md"))
    if verdict != "SIGNAL_SEMANTICS_SYNCED":
        raise RuntimeError(f"signal semantics out of date: {reasons}")
    return summary


def write_signal_semantics_sync_report(summary: dict[str, Any], output: Path) -> None:
    text = f"""# Signal Semantics Status Sync

## Source manifest

{summary["source_manifest"]}

## Repair amount role

{summary["repair_amount_cstr_role"]}

## Invariant residual role

{summary["invariant_residual_cstr_role"]}

## Historical artifact policy

{summary["historical_artifact_policy"]}

## Verdict

{summary["verdict"]}
"""
    _ensure_dir(output.parent)
    output.write_text(text, encoding="utf-8")


def render_readme_status_block(manifest: dict[str, Any]) -> str:
    return f"""{README_START}
## Current Evidence Status

**Current allowed claim:** {manifest["controlling_claim_text"]}

**Expansion status:** Expansion is currently blocked.

**Controlling gates:**
- Practical utility gate: `{manifest["practical_utility_decision"]}`
- Repair signal role gate: `{manifest["repair_signal_role_decision"]}`

**What is supported:**
- TwoTank: calibrated low-coverage refusal has a practically meaningful positive effect.
- CSTR: calibrated low-coverage refusal has a positive but practically weak effect.
- `repair_amount` is correct as a bounds/projection signal but diagnostic-only for CSTR.
- `invariant_residual` is much more informative than repair on CSTR.

**What is not supported:**
- strong general selective simulation
- high-coverage reliability
- safety certification
- product readiness
- autonomous control
- plant-wide digital twin claims
- RSSM or third-system evidence
{README_END}"""


def update_readme_current_status(manifest_path: str | Path, readme_path: str | Path, check: bool = False) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    readme = Path(readme_path)
    current = readme.read_text(encoding="utf-8")
    block = render_readme_status_block(manifest)
    if README_START in current and README_END in current:
        pattern = re.compile(re.escape(README_START) + r".*?" + re.escape(README_END), re.DOTALL)
        updated = pattern.sub(block, current)
    else:
        lines = current.splitlines()
        if lines and lines[0].startswith("# "):
            updated = "\n".join([lines[0], "", block, "", *lines[1:]]) + "\n"
        else:
            updated = block + "\n\n" + current
    stale = updated != current
    if check and stale:
        summary = _readme_sync_summary(manifest, readme, stale=True, check_mode=True)
        _write_json(STATUS_RESULTS_ROOT / "readme_sync" / "readme_status_sync_summary.json", summary)
        write_readme_sync_report(summary, Path("reports/readme_current_status_sync.md"))
        raise RuntimeError("README current status block is stale")
    if not check:
        readme.write_text(updated, encoding="utf-8")
        stale = False
    summary = _readme_sync_summary(manifest, readme, stale=stale, check_mode=check)
    _write_json(STATUS_RESULTS_ROOT / "readme_sync" / "readme_status_sync_summary.json", summary)
    write_readme_sync_report(summary, Path("reports/readme_current_status_sync.md"))
    return summary


def _readme_sync_summary(manifest: dict[str, Any], readme: Path, stale: bool, check_mode: bool) -> dict[str, Any]:
    text = readme.read_text(encoding="utf-8")
    synced = (
        not stale
        and README_START in text
        and manifest["practical_utility_decision"] in text
        and manifest["repair_signal_role_decision"] in text
        and "Expansion is currently blocked" in text
        and "positive but practically weak" in text
        and "diagnostic-only for CSTR" in text
    )
    return {
        "verdict": "README_SYNCED" if synced else "README_STALE",
        "readme": str(readme),
        "check_mode": check_mode,
        "stale": stale,
        "manifest_status_id": manifest["status_id"],
        "controlling_claim": manifest["controlling_claim_text"],
        "expansion_allowed": manifest["expansion_allowed"],
    }


def write_readme_sync_report(summary: dict[str, Any], output: Path) -> None:
    text = f"""# README Current Status Sync

## README

{summary["readme"]}

## Manifest

{summary["manifest_status_id"]}

## Check mode

{summary["check_mode"]}

## Verdict

{summary["verdict"]}
"""
    _ensure_dir(output.parent)
    output.write_text(text, encoding="utf-8")


def check_claim_language(manifest_path: str | Path, paths: list[str | Path]) -> dict[str, Any]:
    _ = _read_json(manifest_path)
    scanned = []
    violations = []
    allowed = []
    for root in [Path(path) for path in paths]:
        candidates = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
        for path in candidates:
            if path.suffix.lower() not in {".md", ".json", ".yaml", ".yml", ".txt"}:
                continue
            if path == Path("reports/claim_language_scan.md"):
                continue
            scanned.append(str(path))
            _scan_claim_file(path, violations, allowed)
    verdict = "CLAIM_LANGUAGE_OK" if not violations else "CLAIM_LANGUAGE_VIOLATIONS"
    result = {
        "verdict": verdict,
        "paths_scanned": scanned,
        "risk_phrases": RISK_PHRASES,
        "violations": violations,
        "allowed_negated_or_forbidden_mentions": allowed,
    }
    out_dir = STATUS_RESULTS_ROOT / "claim_language"
    _ensure_dir(out_dir)
    _write_json(out_dir / "claim_language_scan.json", result)
    write_claim_language_report(result, Path("reports/claim_language_scan.md"))
    if violations:
        raise RuntimeError(f"claim language violations: {violations[:5]}")
    return result


def _scan_claim_file(path: Path, violations: list[dict[str, Any]], allowed: list[dict[str, Any]]) -> None:
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        except json.JSONDecodeError:
            payload = None
        if payload is not None:
            _scan_claim_json_value(path, payload, "", violations, allowed)
            return
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    current_heading = ""
    for lineno, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            current_heading = stripped.lstrip("#").strip().lower()
        elif stripped.startswith("**") and "**" in stripped[2:]:
            current_heading = stripped.split("**", 2)[1].strip(": ").lower()
        low = line.lower()
        for phrase in RISK_PHRASES:
            if phrase not in low:
                continue
            record = {"path": str(path), "line": lineno, "phrase": phrase, "context_heading": current_heading}
            if _claim_mention_is_allowed(low, current_heading):
                allowed.append(record)
            else:
                violations.append(record)


def _scan_claim_json_value(
    path: Path,
    value: Any,
    key_path: str,
    violations: list[dict[str, Any]],
    allowed: list[dict[str, Any]],
) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            next_key = f"{key_path}.{key}" if key_path else str(key)
            _scan_claim_json_value(path, child, next_key, violations, allowed)
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _scan_claim_json_value(path, child, f"{key_path}[{index}]", violations, allowed)
        return
    if not isinstance(value, str):
        return
    low = value.lower()
    context = key_path.lower()
    for phrase in RISK_PHRASES:
        if phrase not in low:
            continue
        record = {"path": str(path), "line": None, "phrase": phrase, "context_heading": context}
        if _claim_mention_is_allowed(low, context):
            allowed.append(record)
        else:
            violations.append(record)


def _claim_mention_is_allowed(line: str, heading: str) -> bool:
    normalized_heading = heading.replace("_", " ").replace("-", " ")
    allowed_heading = any(token in normalized_heading for token in ALLOWED_CONTEXT_HEADINGS)
    negated = any(token in line for token in ["not ", "no ", "forbidden", "blocked", "non-", "does not", "do not", "remains out of scope"])
    return allowed_heading or negated


def write_claim_language_report(result: dict[str, Any], output: Path) -> None:
    violations = pd.DataFrame(result["violations"]) if result["violations"] else pd.DataFrame(columns=["path", "line", "phrase", "context_heading"])
    allowed = pd.DataFrame(result["allowed_negated_or_forbidden_mentions"]) if result["allowed_negated_or_forbidden_mentions"] else pd.DataFrame(columns=["path", "line", "phrase", "context_heading"])
    text = f"""# Claim Language Scan

## Paths scanned

{len(result["paths_scanned"])} files

## Risk phrases

{", ".join(result["risk_phrases"])}

## Violations

{_markdown_table(violations, ["path", "line", "phrase", "context_heading"])}

## Allowed negated/forbidden mentions

{_markdown_table(allowed, ["path", "line", "phrase", "context_heading"])}

## Verdict

{result["verdict"]}
"""
    _ensure_dir(output.parent)
    output.write_text(text, encoding="utf-8")


def make_current_status_decision_gate(
    preconditions_path: str | Path,
    manifest_path: str | Path,
    signal_sync_path: str | Path,
    readme_sync_path: str | Path,
    claim_language_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    preconditions = _read_json(preconditions_path)
    manifest = _read_json(manifest_path)
    signal_sync = _read_json(signal_sync_path)
    readme_sync = _read_json(readme_sync_path)
    claim_language = _read_json(claim_language_path)
    decision = current_status_decision(preconditions, manifest, signal_sync, readme_sync, claim_language)
    allowed_next_action = "MAINTAIN_REPO_AS_WEAK_POSITIVE_BENCHMARK" if decision == "CURRENT_STATUS_SYNCED" else "DO_NOT_EXPAND"
    result = {
        "decision": decision,
        "allowed_next_action": allowed_next_action,
        "allowed_claim": manifest.get("controlling_claim_text"),
        "expansion_allowed": manifest.get("expansion_allowed"),
        "forbidden_claims": manifest.get("forbidden_claims", []),
        "forbidden_next_actions": manifest.get("forbidden_directions", []),
        "inputs": {
            "preconditions": preconditions.get("verdict"),
            "manifest_status_id": manifest.get("status_id"),
            "signal_sync": signal_sync.get("verdict"),
            "readme_sync": readme_sync.get("verdict"),
            "claim_language": claim_language.get("verdict"),
        },
    }
    write_current_status_decision_report(result, Path(output))
    _write_json(Path(output).with_suffix(".json"), result)
    if decision != "CURRENT_STATUS_SYNCED":
        raise RuntimeError(f"current status decision gate failed: {result}")
    return result


def current_status_decision(
    preconditions: dict[str, Any],
    manifest: dict[str, Any],
    signal_sync: dict[str, Any],
    readme_sync: dict[str, Any],
    claim_language: dict[str, Any],
) -> str:
    ok = (
        preconditions.get("verdict") == "READY_FOR_STATUS_SYNC"
        and manifest.get("status_id") == "current_evidence_status_v1"
        and manifest.get("expansion_allowed") is False
        and manifest.get("allowed_next_action") == "UPDATE_SIGNAL_SEMANTICS_ONLY"
        and signal_sync.get("verdict") == "SIGNAL_SEMANTICS_SYNCED"
        and readme_sync.get("verdict") == "README_SYNCED"
        and claim_language.get("verdict") == "CLAIM_LANGUAGE_OK"
    )
    return "CURRENT_STATUS_SYNCED" if ok else "CURRENT_STATUS_STALE_OR_UNSAFE"


def write_current_status_decision_report(result: dict[str, Any], output: Path) -> None:
    inputs = result["inputs"]
    text = f"""# Current Status Decision Gate

## Controlling evidence

{inputs["preconditions"]}

## Manifest status

{inputs["manifest_status_id"]}

## Signal semantics status

{inputs["signal_sync"]}

## README status

{inputs["readme_sync"]}

## Claim-language status

{inputs["claim_language"]}

## Decision

{result["decision"]}

## Allowed next action

{result["allowed_next_action"]}

## Forbidden next actions

{", ".join(result["forbidden_next_actions"])}

## Allowed claim

{result["allowed_claim"]}

## Forbidden claims

{", ".join(result["forbidden_claims"])}
"""
    _ensure_dir(output.parent)
    output.write_text(text, encoding="utf-8")
