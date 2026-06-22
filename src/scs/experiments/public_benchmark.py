from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from scs.data.generate import generate_dataset
from scs.experiments.benchmark_usability import compare_models
from scs.experiments.current_status import (
    README_END,
    README_START,
    _git_dirty_lines,
    _markdown_table,
    _scan_forbidden_runtime_refs,
)
from scs.experiments.technical_note_package import scan_forbidden_claim_language
from scs.metrics.trajectory import rmse
from scs.models.linear_narx import LinearNARXModel


PUBLIC_RESULTS_ROOT = Path("results/public_benchmark_v1_2")
PUBLIC_README_START = "<!-- SCS_PUBLIC_LANDING_START -->"
PUBLIC_README_END = "<!-- SCS_PUBLIC_LANDING_END -->"
USABILITY_START = "<!-- SCS_USABILITY_START -->"
USABILITY_END = "<!-- SCS_USABILITY_END -->"
ALLOWED_CLAIM = "A weak but positive low-coverage result under the frozen protocol."
PUBLIC_HOOK = "A benchmark for testing whether learned dynamical simulators know when to refuse counterfactual predictions."
USER_VALUE = "Plug in a simulator, run OOD/intervention scenarios, and compare false-accept rate versus coverage."
README_OPENING = f"""# Selective Counterfactual Simulation Benchmark

{PUBLIC_HOOK}

{USER_VALUE}

**Current evidence:** weak-positive, synthetic, low-coverage only. Meaningful on TwoTank, weak on CSTR. Not a safety tool.
"""
FORBIDDEN_SURFACE_DIRS = ["api", "frontend", "dashboard", "web", "database"]
PRECONDITION_HASH_ARTIFACTS = [
    "README.md",
    "docs/benchmark_card.md",
    "docs/custom_model_adapter.md",
    "results/current_status/evidence_manifest/current_evidence_manifest.json",
    "results/benchmark_usability/release/benchmark_usability_manifest.json",
    "reports/benchmark_usability_package_check.md",
]
PROTECTED_EVIDENCE_ARTIFACTS = [
    "results/current_status/evidence_manifest/current_evidence_manifest.json",
    "results/calibrated_two_tank/low_coverage_summary.csv",
    "results/calibrated_cstr/low_coverage_summary.csv",
    "results/benchmark_usability/release/benchmark_usability_manifest.json",
]
REQUIRED_README_SECTIONS = [
    "## Why this exists",
    "## Quickstart",
    "## Reproduce the main TwoTank result",
    "## Plug in your own simulator",
    "## Main result",
    "## What this does not claim",
    "## Repository map",
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


def load_public_config(path: str | Path) -> dict[str, Any]:
    config = _read_yaml(path)
    required = {
        "package_id",
        "source_commit",
        "source_tag",
        "allowed_claim",
        "expansion_policy",
        "required_public_hook",
        "required_user_value",
        "source_artifacts",
        "forbidden_claims",
        "forbidden_research_expansion",
        "public_outputs",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"missing public benchmark config keys: {missing}")
    if config["package_id"] != "public_benchmark_v1_2":
        raise ValueError("unexpected package_id")
    if config["allowed_claim"]["text"] != ALLOWED_CLAIM:
        raise ValueError("allowed claim changed")
    policy = config["expansion_policy"]
    if policy.get("scientific_expansion_allowed") is not False:
        raise ValueError("scientific expansion must be false")
    if policy.get("usability_expansion_allowed") is not True:
        raise ValueError("usability expansion must be true")
    if policy.get("public_packaging_allowed") is not True:
        raise ValueError("public packaging must be true")
    if config["required_public_hook"]["text"] != PUBLIC_HOOK:
        raise ValueError("public hook changed")
    if config["required_user_value"]["text"] != USER_VALUE:
        raise ValueError("user value changed")
    forbidden = yaml.safe_dump(config["forbidden_research_expansion"]).lower()
    for token in ["rssm", "heat_exchanger", "third-system", "product/api/frontend"]:
        if token not in forbidden:
            raise ValueError(f"forbidden research expansion must include {token}")
    return config


def verify_public_benchmark_preconditions(
    config_path: str | Path,
    output: str | Path,
    report_output: str | Path = "reports/public_benchmark_precondition_check.md",
) -> dict[str, Any]:
    config = load_public_config(config_path)
    out_dir = Path(output)
    _ensure_dir(out_dir)
    source_artifacts = config["source_artifacts"]
    required_artifacts = {
        **source_artifacts,
        "readme": "README.md",
        "benchmark_usability_package_check": "reports/benchmark_usability_package_check.md",
    }
    missing = [path for path in required_artifacts.values() if not Path(path).exists() or Path(path).stat().st_size == 0]
    usability_check = _read_json("results/benchmark_usability/package_check.json")
    current_manifest = _read_json(source_artifacts["current_manifest"])
    dependency_scan = _scan_forbidden_runtime_refs([Path("src"), Path("scripts"), Path("examples")])
    forbidden_refs = _forbidden_evidence_refs(source_artifacts)
    surface_dirs = [path for path in FORBIDDEN_SURFACE_DIRS if Path(path).exists()]
    hashes = {
        path: {"sha256": _sha256(path), "bytes": Path(path).stat().st_size}
        for path in PRECONDITION_HASH_ARTIFACTS
        if Path(path).exists()
    }
    _write_json(out_dir / "source_artifact_hashes.json", {"artifacts": hashes})
    reasons: list[str] = []
    if missing:
        reasons.append(f"missing source artifacts: {missing}")
    if current_manifest.get("controlling_claim_text") != ALLOWED_CLAIM:
        reasons.append("current allowed claim is not the weak low-coverage benchmark claim")
    if current_manifest.get("expansion_allowed") is not False:
        reasons.append("current evidence manifest allows expansion")
    if config["expansion_policy"]["scientific_expansion_allowed"] is not False:
        reasons.append("scientific expansion is allowed")
    if usability_check.get("verdict") != "BENCHMARK_USABILITY_PACKAGE_ACCEPTED":
        reasons.append("benchmark usability package is not accepted")
    if dependency_scan["old_repo_runtime_import_hits"] or dependency_scan["path_hack_hits"]:
        reasons.append("forbidden runtime dependency/path scan failed")
    if forbidden_refs:
        reasons.extend(forbidden_refs)
    if surface_dirs:
        reasons.append(f"forbidden product/API/frontend directories exist: {surface_dirs}")
    verdict = "READY_FOR_PUBLIC_BENCHMARK_PACKAGING" if not reasons else "NOT_READY"
    result = {
        "package_id": config["package_id"],
        "working_tree_dirty": bool(_git_dirty_lines()),
        "dirty_state": _git_dirty_lines(),
        "allowed_claim": config["allowed_claim"]["text"],
        "current_manifest_claim": current_manifest.get("controlling_claim_text"),
        "current_manifest_expansion_allowed": current_manifest.get("expansion_allowed"),
        "scientific_expansion_allowed": config["expansion_policy"]["scientific_expansion_allowed"],
        "public_packaging_allowed": config["expansion_policy"]["public_packaging_allowed"],
        "benchmark_usability_package": usability_check.get("verdict"),
        "required_source_artifacts": [
            {"name": key, "path": path, "exists": path not in missing}
            for key, path in required_artifacts.items()
        ],
        "forbidden_dependency_scan": dependency_scan,
        "forbidden_evidence_refs": forbidden_refs,
        "forbidden_surface_dirs": surface_dirs,
        "source_artifact_hash_manifest": str(out_dir / "source_artifact_hashes.json"),
        "verdict": verdict,
        "reasons": reasons,
    }
    _write_json(out_dir / "precondition_check.json", result)
    write_public_precondition_report(result, report_output)
    if verdict != "READY_FOR_PUBLIC_BENCHMARK_PACKAGING":
        raise RuntimeError(f"public benchmark preconditions failed: {reasons}")
    return result


def _forbidden_evidence_refs(source_artifacts: dict[str, str]) -> list[str]:
    source_text = yaml.safe_dump(source_artifacts).lower()
    refs = []
    if "heat_exchanger" in source_text:
        refs.append("heat_exchanger referenced as evidence")
    if "rssm" in source_text:
        refs.append("RSSM referenced as evidence")
    return refs


def write_public_precondition_report(result: dict[str, Any], output: str | Path) -> None:
    artifacts = pd.DataFrame(result["required_source_artifacts"])
    scan = result["forbidden_dependency_scan"]
    text = f"""# Public Benchmark Preconditions

## Working tree

Dirty: {result["working_tree_dirty"]}

## Current evidence status

Allowed claim: {result["allowed_claim"]}

Current manifest claim: {result["current_manifest_claim"]}

Current manifest expansion allowed: {result["current_manifest_expansion_allowed"]}

Benchmark usability package: {result["benchmark_usability_package"]}

## Expansion policy

Scientific expansion allowed: {result["scientific_expansion_allowed"]}

Public packaging allowed: {result["public_packaging_allowed"]}

## Required source artifacts

{_markdown_table(artifacts, ["name", "path", "exists"])}

## Forbidden dependency scan

Old repo runtime import hits: {scan["old_repo_runtime_import_hits"] or "none"}

Path hack hits: {scan["path_hack_hits"] or "none"}

Forbidden evidence refs: {result["forbidden_evidence_refs"] or "none"}

Forbidden surface dirs: {result["forbidden_surface_dirs"] or "none"}

## Source artifact hashes

{result["source_artifact_hash_manifest"]}

## Verdict

{result["verdict"]}
"""
    target = Path(output)
    _ensure_dir(target.parent)
    target.write_text(text, encoding="utf-8")


def run_smoke_demo(output: str | Path) -> dict[str, Any]:
    out_dir = Path(output)
    _ensure_dir(out_dir)
    dataset = generate_dataset("two_tank", n_train=24, n_id_test=6, n_ood_test=6, horizon=12, dt=0.1, seed=123)
    model = LinearNARXModel()
    model.fit(dataset["train"])
    rows = []
    for split in ["id_test", "ood_action_magnitude", "ood_inflow_spike", "ood_combined"]:
        batch = dataset[split]
        errors = []
        for idx in range(batch.n_trajectories):
            pred = model.predict_rollout(batch.states[idx, 0], batch.actions[idx], batch.disturbances[idx])
            errors.append(rmse(pred, batch.states[idx]))
        rows.append({"split": split, "mean_rmse": float(np.mean(errors)), "n_scenarios": len(errors)})
    table = pd.DataFrame(rows)
    _plot_smoke_demo(table, out_dir / "smoke_demo_plot.png")
    summary = {
        "verdict": "SMOKE_DEMO_BUILT",
        "is_smoke_only": True,
        "is_main_evidence": False,
        "system_id": "two_tank",
        "model_id": model.model_id,
        "n_train": dataset["train"].n_trajectories,
        "outputs": {
            "summary": str(out_dir / "smoke_demo_summary.json"),
            "report": str(out_dir / "smoke_demo_report.md"),
            "plot": str(out_dir / "smoke_demo_plot.png"),
        },
    }
    _write_json(out_dir / "smoke_demo_summary.json", summary)
    write_smoke_demo_report(table, summary, out_dir / "smoke_demo_report.md")
    return summary


def _plot_smoke_demo(table: pd.DataFrame, output: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(table["split"], table["mean_rmse"], color=["#3b82f6", "#f97316", "#14b8a6", "#8b5cf6"])
    ax.set_title("Smoke demo trajectory error")
    ax.set_ylabel("Mean RMSE")
    ax.set_xlabel("Split")
    ax.tick_params(axis="x", rotation=20)
    ax.text(
        0.02,
        0.95,
        "Smoke-only pipeline check",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    fig.tight_layout()
    target = Path(output)
    _ensure_dir(target.parent)
    fig.savefig(target, dpi=150)
    plt.close(fig)


def write_smoke_demo_report(table: pd.DataFrame, summary: dict[str, Any], output: str | Path) -> None:
    text = f"""# Smoke Demo Report

This smoke demo checks that the benchmark pipeline runs; it is not the full evidence reproduction.

## Scope

System: {summary["system_id"]}

Model: {summary["model_id"]}

## Split RMSE

{_markdown_table(table, ["split", "mean_rmse", "n_scenarios"])}

## Claim boundary

This smoke output is not evidence for the current supported claim.

## Verdict

{summary["verdict"]}
"""
    target = Path(output)
    _ensure_dir(target.parent)
    target.write_text(text, encoding="utf-8")


def reproduce_main_twotank_result(output: str | Path) -> dict[str, Any]:
    out_dir = Path(output)
    _ensure_dir(out_dir)
    source = Path("results/calibrated_two_tank/low_coverage_summary.csv")
    table = twotank_main_result_table(source)
    table.to_csv(out_dir / "twotank_main_result.csv", index=False)
    _plot_twotank_reproduction(table, out_dir / "twotank_main_result.png")
    summary = {
        "verdict": "TWOTANK_MAIN_RESULT_REPRODUCED",
        "source_artifact": str(source),
        "is_reproduction": True,
        "current_manifest_modified": False,
        "coverage_0_05_margin": float(table[np.isclose(table["coverage"], 0.05)].iloc[0]["absolute_margin"]),
        "coverage_0_10_margin": float(table[np.isclose(table["coverage"], 0.10)].iloc[0]["absolute_margin"]),
        "outputs": {
            "table": str(out_dir / "twotank_main_result.csv"),
            "plot": str(out_dir / "twotank_main_result.png"),
            "summary": str(out_dir / "twotank_reproduction_summary.json"),
            "report": str(out_dir / "twotank_reproduction_report.md"),
        },
    }
    _write_json(out_dir / "twotank_reproduction_summary.json", summary)
    write_twotank_reproduction_report(table, summary, out_dir / "twotank_reproduction_report.md")
    return summary


def twotank_main_result_table(source: str | Path = "results/calibrated_two_tank/low_coverage_summary.csv") -> pd.DataFrame:
    low = pd.read_csv(source)
    rows = []
    for coverage in [0.05, 0.10]:
        match = low[np.isclose(low["coverage"].astype(float), coverage)]
        if match.empty:
            raise ValueError(f"missing TwoTank low-coverage row {coverage}")
        row = match.iloc[0]
        rows.append(
            {
                "system_id": "two_tank",
                "coverage": float(row["coverage"]),
                "baseline_far": float(row["baseline_far"]),
                "calibrated_far": float(row["calibrated_far"]),
                "absolute_margin": float(row["margin"]),
                "source_artifact": str(source),
                "is_reproduction": True,
            }
        )
    return pd.DataFrame(rows)


def _plot_twotank_reproduction(table: pd.DataFrame, output: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    width = 0.34
    x = np.arange(len(table))
    ax.bar(x - width / 2, table["baseline_far"], width, label="Baseline FAR", color="#64748b")
    ax.bar(x + width / 2, table["calibrated_far"], width, label="Calibrated FAR", color="#2563eb")
    for idx, margin in enumerate(table["absolute_margin"]):
        ax.text(idx, max(table.iloc[idx]["baseline_far"], table.iloc[idx]["calibrated_far"]) + 0.02, f"margin {margin:.3f}", ha="center", fontsize=9)
    ax.set_title("TwoTank low-coverage reproduction")
    ax.set_ylabel("False accept rate")
    ax.set_xlabel("Coverage")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{value:.2f}" for value in table["coverage"]])
    ax.legend()
    fig.tight_layout()
    target = Path(output)
    _ensure_dir(target.parent)
    fig.savefig(target, dpi=150)
    plt.close(fig)


def write_twotank_reproduction_report(table: pd.DataFrame, summary: dict[str, Any], output: str | Path) -> None:
    text = f"""# TwoTank Main Result Reproduction

## Source artifact

{summary["source_artifact"]}

## Result

{_markdown_table(table, ["system_id", "coverage", "baseline_far", "calibrated_far", "absolute_margin", "source_artifact", "is_reproduction"])}

## Interpretation

The TwoTank low-coverage margins are nonzero: {summary["coverage_0_05_margin"]:.6f} at coverage 0.05 and {summary["coverage_0_10_margin"]:.6f} at coverage 0.10.

This reproduces an existing frozen artifact and does not change the current evidence manifest.

## Claim boundary

This is weak-positive low-coverage synthetic benchmark evidence only.

## Verdict

{summary["verdict"]}
"""
    target = Path(output)
    _ensure_dir(target.parent)
    target.write_text(text, encoding="utf-8")


def build_readme_main_figure(
    manifest_path: str | Path,
    output: str | Path,
    report_output: str | Path = "reports/readme_main_figure_report.md",
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    rows = []
    for system_id, label in [("two_tank", "TwoTank"), ("cstr", "CSTR")]:
        item = manifest["systems"][system_id]
        rows.append({"system": label, "coverage": 0.05, "margin": float(item["coverage_0_05_margin"])})
        rows.append({"system": label, "coverage": 0.10, "margin": float(item["coverage_0_10_margin"])})
    table = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    colors = ["#2563eb" if system == "TwoTank" else "#f97316" for system in table["system"]]
    labels = [f"{row.system}\ncoverage {row.coverage:.2f}" for row in table.itertuples()]
    ax.bar(labels, table["margin"], color=colors)
    ax.set_title("Low-coverage false-accept reduction")
    ax.set_ylabel("FAR reduction margin")
    ax.set_xlabel("System and coverage")
    ax.text(
        0.02,
        0.95,
        "Weak-positive synthetic benchmark result; not safety evidence.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    for idx, value in enumerate(table["margin"]):
        ax.text(idx, value + 0.006, f"{value:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0, max(table["margin"]) * 1.35)
    fig.tight_layout()
    target = Path(output)
    _ensure_dir(target.parent)
    fig.savefig(target, dpi=160)
    plt.close(fig)
    figure_manifest = {
        "verdict": "README_MAIN_FIGURE_BUILT",
        "source_manifest": str(manifest_path),
        "output": str(output),
        "title": "Low-coverage false-accept reduction",
        "subtitle": "Weak-positive synthetic benchmark result; not safety evidence.",
        "rows": rows,
    }
    if target.resolve() == Path("docs/figures/readme_low_coverage_result.png").resolve():
        _write_json(PUBLIC_RESULTS_ROOT / "readme_figure_manifest.json", figure_manifest)
    write_readme_main_figure_report(figure_manifest, report_output)
    return figure_manifest


def write_readme_main_figure_report(figure_manifest: dict[str, Any], output: str | Path) -> None:
    table = pd.DataFrame(figure_manifest["rows"])
    text = f"""# README Main Figure Report

## Source manifest

{figure_manifest["source_manifest"]}

## Figure

{figure_manifest["output"]}

## Title

{figure_manifest["title"]}

## Figure data

{_markdown_table(table, ["system", "coverage", "margin"])}

## Caption

Weak-positive synthetic benchmark result; not safety evidence.

## Verdict

{figure_manifest["verdict"]}
"""
    target = Path(output)
    _ensure_dir(target.parent)
    target.write_text(text, encoding="utf-8")


def render_public_landing(config: dict[str, Any]) -> str:
    return f"""{README_OPENING}
{PUBLIC_README_START}

## Why this exists

Learned simulators can look accurate in-distribution and still fail under counterfactual intervention shift. This benchmark asks a narrower question: can a simulator or refusal judge rank scenarios by answerability and abstain on risky ones?

## Quickstart

```bash
pip install -e ".[dev]"
pytest -q
python scripts/run_smoke_demo.py --output results/smoke_demo
python scripts/reproduce_main_twotank_result.py --output results/reproduce_twotank
python scripts/compare_models.py --config configs/experiments/calibrated_two_tank.yaml --models hold_last linear_narx mlp_state_space --output results/model_comparison
```

## Reproduce the main TwoTank result

```bash
python scripts/reproduce_main_twotank_result.py --output results/reproduce_twotank
```

This reads the frozen TwoTank low-coverage artifact and writes a small reproduction table and figure with the nonzero margins.

## Plug in your own simulator

Start from `examples/my_model_template.py` or the runnable example in `examples/custom_model_example.py`, then compare locally:

```bash
python examples/custom_model_example.py --output results/custom_model_example
python scripts/compare_models.py --config configs/experiments/calibrated_two_tank.yaml --models linear_narx mlp_state_space --custom-model examples/custom_model_example.py:DampedLinearUserModel --output results/model_comparison_custom
```

Custom model outputs are local comparison results only; they are not added to the frozen evidence claim.

## Main result

![Low-coverage false-accept reduction](docs/figures/readme_low_coverage_result.png)

The effect is meaningful on TwoTank and weak on CSTR; this is low-coverage synthetic benchmark evidence only.

Current allowed claim: {config["allowed_claim"]["text"]}

## What this does not claim

This benchmark does not claim simulator safety, product readiness, broad simulator reliability, high-coverage reliability, plant-wide deployment, autonomous control, RSSM evidence, heat-exchanger evidence, or third-system evidence.

## Repository map

- `src/scs/systems/`: synthetic systems.
- `src/scs/models/`: benchmark model interface and built-in baselines.
- `src/scs/validators/`: refusal/risk signals and judges.
- `src/scs/metrics/`: trajectory, event, and risk-coverage metrics.
- `src/scs/experiments/`: reproducible experiment and packaging logic.
- `configs/`: frozen experiment, audit, and status configs.
- `docs/`: benchmark card, task definition, failure gallery, and reproducibility notes.
- `reports/` and `results/`: generated evidence artifacts.

{PUBLIC_README_END}"""


def update_readme_public_landing(config_path: str | Path, readme_path: str | Path, check: bool = False) -> dict[str, Any]:
    config = load_public_config(config_path)
    readme = Path(readme_path)
    current = readme.read_text(encoding="utf-8")
    public_block = render_public_landing(config)
    preserved_blocks = _extract_status_blocks(current)
    historical = _strip_existing_public_readme(current)
    detailed = "\n\n## Detailed generated status blocks\n\n" + "\n\n".join(block for block in preserved_blocks if block.strip())
    updated = public_block + detailed + "\n\n" + historical.lstrip()
    stale = updated != current
    if check and stale:
        raise RuntimeError("README public landing is stale")
    if not check:
        readme.write_text(updated, encoding="utf-8")
        stale = False
    summary = _readme_public_summary(readme, stale=stale, check=check)
    _write_json(PUBLIC_RESULTS_ROOT / "readme_public_landing_check.json", summary)
    if summary["verdict"] != "README_PUBLIC_LANDING_SYNCED":
        raise RuntimeError(f"README public landing check failed: {summary}")
    return summary


def _extract_status_blocks(text: str) -> list[str]:
    blocks = []
    for start, end in [(README_START, README_END), (USABILITY_START, USABILITY_END)]:
        if start in text and end in text:
            match = re.search(re.escape(start) + r".*?" + re.escape(end), text, flags=re.DOTALL)
            if match:
                blocks.append(match.group(0))
    return blocks


def _strip_existing_public_readme(text: str) -> str:
    if PUBLIC_README_START in text and PUBLIC_README_END in text:
        text = re.sub(r"^.*?" + re.escape(PUBLIC_README_END), "", text, flags=re.DOTALL)
    else:
        text = re.sub(re.escape(PUBLIC_README_START) + r".*?" + re.escape(PUBLIC_README_END), "", text, flags=re.DOTALL)
    text = re.sub(re.escape(README_START) + r".*?" + re.escape(README_END), "", text, flags=re.DOTALL)
    text = re.sub(re.escape(USABILITY_START) + r".*?" + re.escape(USABILITY_END), "", text, flags=re.DOTALL)
    text = re.sub(r"(?m)^## Detailed generated status blocks\s*", "", text)
    lines = text.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    return "\n".join(lines).strip() + "\n"


def _readme_public_summary(readme: Path, stale: bool, check: bool) -> dict[str, Any]:
    text = readme.read_text(encoding="utf-8")
    missing = [section for section in REQUIRED_README_SECTIONS if section not in text]
    required_snippets = [
        PUBLIC_HOOK,
        USER_VALUE,
        "weak-positive, synthetic, low-coverage only",
        "Not a safety tool",
        "python scripts/run_smoke_demo.py --output results/smoke_demo",
        "python scripts/reproduce_main_twotank_result.py --output results/reproduce_twotank",
        "examples/custom_model_example.py",
        "docs/figures/readme_low_coverage_result.png",
        "weak on CSTR",
        "does not claim simulator safety",
    ]
    missing.extend(snippet for snippet in required_snippets if snippet not in text)
    starts_correctly = text.startswith(README_OPENING)
    if not starts_correctly:
        missing.append("required opening")
    return {
        "verdict": "README_PUBLIC_LANDING_SYNCED" if not stale and not missing else "README_PUBLIC_LANDING_STALE",
        "readme": str(readme),
        "check_mode": check,
        "stale": stale,
        "missing": missing,
        "starts_with_public_hook": starts_correctly,
    }


def write_quickstart_doc(output: str | Path = "docs/quickstart.md") -> None:
    text = """# Quickstart

## Install

```bash
pip install -e ".[dev]"
pytest -q
```

## Smoke demo

```bash
python scripts/run_smoke_demo.py --output results/smoke_demo
```

This checks that the benchmark pipeline runs. It is not the full evidence reproduction.

## Main TwoTank reproduction

```bash
python scripts/reproduce_main_twotank_result.py --output results/reproduce_twotank
```

## Local model comparison

```bash
python scripts/compare_models.py --config configs/experiments/calibrated_two_tank.yaml --models hold_last linear_narx mlp_state_space --output results/model_comparison
```

## Custom model comparison

```bash
python scripts/compare_models.py --config configs/experiments/calibrated_two_tank.yaml --models linear_narx mlp_state_space --custom-model examples/custom_model_example.py:DampedLinearUserModel --output results/model_comparison_custom
```

## Claim boundary

The current claim remains weak-positive and low-coverage only.
"""
    target = Path(output)
    _ensure_dir(target.parent)
    target.write_text(text, encoding="utf-8")


def write_refusal_benchmark_task(output: str | Path = "docs/tasks/refusal_benchmark_task.md") -> None:
    text = """# Task: Refusal Benchmark for Counterfactual Simulators

## Task summary

Given a simulator and counterfactual intervention scenarios, rank scenarios by answerability and accept only the lowest-risk fraction.

## Input

Synthetic train, calibration, and test trajectory batches with states, actions, disturbances, scenario labels, and intervention splits.

## Output

Per-scenario rollout predictions, error metrics, refusal/risk scores, and risk-coverage tables.

## Systems

Current evidence uses TwoTank and CSTR. TwoTank is stronger than CSTR.

## Scenario types

Normal policy, held-out action magnitude, step changes, inflow/feed spikes, degradation, combined interventions, and CSTR unsafe event scenarios.

## Models

Built-in baselines include hold-last, Linear NARX, and MLP state-space. User models can be added through the local adapter.

## Refusal signals

Support distance, uncertainty, disagreement, invariant residual, and repair amount. repair_amount is diagnostic-only for CSTR; invariant_residual is more informative there.

## Primary metric

False accept rate at fixed coverage.

## False accept definition

A false accept occurs when a judge accepts a scenario but the simulator rollout is bad under the configured error/event label.

## Coverage definition

Coverage is the fraction of scenarios accepted by the judge.

## Baselines

Support-only, uncertainty-only, disagreement-only, invariant-only, repair-only, random, oracle diagnostic, and calibrated/rank-based judges.

## Current best known result

Weak-positive at low coverage; TwoTank stronger than CSTR.

## How to submit/evaluate your own model locally

Implement `fit(train_batch)` and `predict_rollout(initial_state, actions, disturbances)`, then run `scripts/compare_models.py` with `--custom-model path.py:ClassName`.

## Fair comparison rules

Use the same splits, do not tune on test labels, report custom runs as local-only, and do not mix custom outputs into the frozen evidence claim.

## Non-goals

This is not production validation, safety certification, product readiness, autonomous control, plant-wide simulation, high-coverage reliability, or RSSM/third-system evidence.
"""
    target = Path(output)
    _ensure_dir(target.parent)
    target.write_text(text, encoding="utf-8")


def build_failure_gallery(config_path: str | Path, output: str | Path, figure_dir: str | Path) -> dict[str, Any]:
    _ = load_public_config(config_path)
    out_path = Path(output)
    fig_dir = Path(figure_dir)
    _ensure_dir(fig_dir)
    examples = _select_failure_examples()
    for example in examples:
        _plot_failure_example(example, fig_dir / f"{example['example_id']}.png")
        example["figure"] = str(fig_dir / f"{example['example_id']}.png")
    write_failure_gallery_markdown(examples, out_path)
    manifest = {
        "verdict": "FAILURE_GALLERY_BUILT",
        "source_artifacts": [
            "results/calibrated_two_tank/test_table.csv",
            "results/calibrated_cstr/test_table.csv",
        ],
        "examples": examples,
    }
    if out_path.resolve() == Path("docs/failure_gallery.md").resolve():
        _write_json(PUBLIC_RESULTS_ROOT / "failure_gallery_manifest.json", manifest)
        write_failure_gallery_report(manifest, "reports/failure_gallery_report.md")
    return manifest


def _select_failure_examples() -> list[dict[str, Any]]:
    two = _mark_accepted(pd.read_csv("results/calibrated_two_tank/test_table.csv"), "risk_calibration_selected_candidate_ranker", 0.05)
    cstr_support = _mark_accepted(pd.read_csv("results/calibrated_cstr/test_table.csv"), "risk_support_only", 0.05)
    cstr_rank = _mark_accepted(pd.read_csv("results/calibrated_cstr/test_table.csv"), "risk_rank_normalized_linear", 0.05)
    examples = [
        _row_to_example(
            "example_1_accepted_good",
            "Example 1: Accepted good scenario",
            two[(two["accepted"]) & (~two["bad_rmse_label"])].sort_values("rmse").iloc[0],
            "calibration_selected_candidate_ranker",
            0.05,
        ),
        _row_to_example(
            "example_2_correctly_rejected_bad",
            "Example 2: Correctly rejected bad scenario",
            two[(~two["accepted"]) & (two["bad_rmse_label"])].sort_values("rmse", ascending=False).iloc[0],
            "calibration_selected_candidate_ranker",
            0.05,
        ),
        _row_to_example(
            "example_3_false_accept_cstr",
            "Example 3: False accept",
            cstr_support[(cstr_support["accepted"]) & (cstr_support["bad_rmse_label"])].sort_values("rmse", ascending=False).iloc[0],
            "support_only",
            0.05,
        ),
        _row_to_example(
            "example_4_cstr_within_bound_dynamic_failure",
            "Example 4: CSTR within-bound dynamic failure",
            cstr_rank[(~cstr_rank["accepted"]) & (cstr_rank["bad_rmse_label"]) & (cstr_rank["repair_amount"] <= 1e-12)].sort_values("rmse", ascending=False).iloc[0],
            "rank_normalized_linear",
            0.05,
        ),
        _row_to_example(
            "example_5_invariant_residual_helps",
            "Example 5: Invariant residual helps",
            cstr_rank[(~cstr_rank["accepted"]) & (cstr_rank["bad_rmse_label"]) & (cstr_rank["invariant_residual"] > 0.05)].sort_values("invariant_residual", ascending=False).iloc[0],
            "rank_normalized_linear",
            0.05,
        ),
    ]
    return examples


def _mark_accepted(df: pd.DataFrame, risk_col: str, coverage: float) -> pd.DataFrame:
    marked = df.copy()
    accepted_count = int(np.ceil(len(marked) * coverage))
    accepted = marked[risk_col].sort_values(kind="mergesort").index[:accepted_count]
    marked["accepted"] = False
    marked.loc[accepted, "accepted"] = True
    marked["risk_column"] = risk_col
    marked["risk_value"] = marked[risk_col]
    return marked


def _row_to_example(example_id: str, title: str, row: pd.Series, judge: str, coverage: float) -> dict[str, Any]:
    false_accept = bool(row["accepted"] and row["bad_rmse_label"])
    return {
        "example_id": example_id,
        "title": title,
        "source_artifact": f"results/calibrated_{row['system_id']}/test_table.csv" if row["system_id"] in {"two_tank", "cstr"} else "unknown",
        "system": str(row["system_id"]),
        "model": str(row["model_id"]),
        "scenario_id": str(row["scenario_id"]),
        "scenario_type": str(row["scenario_type"]),
        "judge": judge,
        "coverage": float(coverage),
        "rmse": float(row["rmse"]),
        "accepted": bool(row["accepted"]),
        "decision": "accepted" if row["accepted"] else "refused",
        "false_accept": false_accept,
        "bad_rollout": bool(row["bad_rmse_label"]),
        "key_signal_values": {
            "support_distance": float(row["support_distance"]),
            "uncertainty_score": float(row["uncertainty_score"]),
            "disagreement_score": float(row["disagreement_score"]),
            "invariant_residual": float(row["invariant_residual"]),
            "repair_amount": float(row["repair_amount"]),
            "risk_value": float(row["risk_value"]),
        },
    }


def _plot_failure_example(example: dict[str, Any], output: str | Path) -> None:
    signals = example["key_signal_values"]
    labels = ["support", "uncertainty", "disagreement", "invariant", "repair"]
    values = [
        signals["support_distance"],
        signals["uncertainty_score"],
        signals["disagreement_score"],
        signals["invariant_residual"],
        signals["repair_amount"],
    ]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.bar(labels, values, color=["#64748b", "#3b82f6", "#8b5cf6", "#14b8a6", "#f97316"])
    ax.set_title(example["title"])
    ax.set_ylabel("Signal value")
    ax.text(0.02, 0.95, f"{example['system']} / {example['model']} / {example['decision']}", transform=ax.transAxes, ha="left", va="top", fontsize=9)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    target = Path(output)
    _ensure_dir(target.parent)
    fig.savefig(target, dpi=140)
    plt.close(fig)


def write_failure_gallery_markdown(examples: list[dict[str, Any]], output: str | Path) -> None:
    by_id = {example["example_id"]: example for example in examples}
    sections = [
        "# Failure Gallery",
        "## Why this gallery exists",
        "These examples make the benchmark concrete. They are selected from actual calibrated test-table artifacts, not fabricated cases.",
    ]
    ordered = [
        ("example_1_accepted_good", "## Example 1: Accepted good scenario"),
        ("example_2_correctly_rejected_bad", "## Example 2: Correctly rejected bad scenario"),
        ("example_3_false_accept_cstr", "## Example 3: False accept"),
        ("example_4_cstr_within_bound_dynamic_failure", "## Example 4: CSTR within-bound dynamic failure"),
        ("example_5_invariant_residual_helps", "## Example 5: Invariant residual helps"),
    ]
    for key, heading in ordered:
        example = by_id[key]
        signals = example["key_signal_values"]
        sections.append(heading)
        sections.append(
            f"""System: {example["system"]}

Model: {example["model"]}

Scenario type: {example["scenario_type"]}

Judge: {example["judge"]}

Coverage: {example["coverage"]:.2f}

RMSE: {example["rmse"]:.6f}

Decision: {example["decision"]}

False accept status: {example["false_accept"]}

Key signal values: support={signals["support_distance"]:.6f}, uncertainty={signals["uncertainty_score"]:.6f}, disagreement={signals["disagreement_score"]:.6f}, invariant={signals["invariant_residual"]:.6f}, repair={signals["repair_amount"]:.6f}, risk={signals["risk_value"]:.6f}

Source artifact: {example["source_artifact"]}

![{example["title"]}]({example["figure"]})
"""
        )
    sections.append(
        """## What these examples show

They show that accepted-good, correctly-rejected-bad, and false-accept cases all exist in the frozen artifacts. They also show the CSTR repair-signal blind spot: repair can be zero while invariant residual is informative.

## What these examples do not prove

They do not prove simulator safety, product readiness, broad reliability, or high-coverage performance.
"""
    )
    target = Path(output)
    _ensure_dir(target.parent)
    target.write_text("\n\n".join(sections), encoding="utf-8")


def write_failure_gallery_report(manifest: dict[str, Any], output: str | Path) -> None:
    examples = pd.DataFrame(
        [
            {
                "example_id": item["example_id"],
                "system": item["system"],
                "model": item["model"],
                "decision": item["decision"],
                "false_accept": item["false_accept"],
                "rmse": item["rmse"],
                "repair_amount": item["key_signal_values"]["repair_amount"],
                "invariant_residual": item["key_signal_values"]["invariant_residual"],
            }
            for item in manifest["examples"]
        ]
    )
    text = f"""# Failure Gallery Report

## Source artifacts

{", ".join(manifest["source_artifacts"])}

## Examples

{_markdown_table(examples, ["example_id", "system", "model", "decision", "false_accept", "rmse", "repair_amount", "invariant_residual"])}

## Verdict

{manifest["verdict"]}
"""
    target = Path(output)
    _ensure_dir(target.parent)
    target.write_text(text, encoding="utf-8")


def check_public_benchmark_package(config_path: str | Path) -> dict[str, Any]:
    config = load_public_config(config_path)
    temp_root = PUBLIC_RESULTS_ROOT / "package_check_runs"
    pre = verify_public_benchmark_preconditions(config_path, PUBLIC_RESULTS_ROOT / "preconditions")
    smoke = run_smoke_demo(temp_root / "smoke_demo")
    reproduction = reproduce_main_twotank_result(temp_root / "reproduce_twotank")
    readme_check = update_readme_public_landing(config_path, "README.md", check=True)
    custom_run = subprocess.run(
        [sys.executable, "examples/custom_model_example.py", "--output", str(temp_root / "custom_model_example")],
        check=False,
        capture_output=True,
        text=True,
    )
    builtin_compare = compare_models(
        "configs/experiments/calibrated_two_tank.yaml",
        ["hold_last", "linear_narx"],
        temp_root / "model_comparison",
    )
    custom_compare = compare_models(
        "configs/experiments/calibrated_two_tank.yaml",
        ["linear_narx"],
        temp_root / "model_comparison_custom",
        custom_model="examples/custom_model_example.py:DampedLinearUserModel",
    )
    claim_scan = scan_forbidden_claim_language(
        [
            "README.md",
            "docs",
            "reports",
        ],
        config["forbidden_claims"],
    )
    source_hashes = _read_json(PUBLIC_RESULTS_ROOT / "preconditions" / "source_artifact_hashes.json")["artifacts"]
    hash_mismatches = [
        path
        for path, item in source_hashes.items()
        if Path(path).exists()
        and path in PROTECTED_EVIDENCE_ARTIFACTS
        and _sha256(path) != item["sha256"]
    ]
    required_files = [
        "results/public_benchmark_v1_2/preconditions/precondition_check.json",
        "results/smoke_demo/smoke_demo_summary.json",
        "results/smoke_demo/smoke_demo_report.md",
        "results/smoke_demo/smoke_demo_plot.png",
        "results/reproduce_twotank/twotank_main_result.csv",
        "results/reproduce_twotank/twotank_main_result.png",
        "results/reproduce_twotank/twotank_reproduction_summary.json",
        "results/reproduce_twotank/twotank_reproduction_report.md",
        "docs/figures/readme_low_coverage_result.png",
        "results/public_benchmark_v1_2/readme_figure_manifest.json",
        "docs/tasks/refusal_benchmark_task.md",
        "docs/failure_gallery.md",
        "results/public_benchmark_v1_2/failure_gallery_manifest.json",
        "examples/my_model_template.py",
        "examples/custom_model_example.py",
        "docs/custom_model_adapter.md",
        ".github/workflows/ci.yml",
    ]
    missing = [path for path in required_files if not Path(path).exists() or Path(path).stat().st_size == 0]
    reasons = []
    if pre.get("verdict") != "READY_FOR_PUBLIC_BENCHMARK_PACKAGING":
        reasons.append("preconditions are not ready")
    if smoke.get("verdict") != "SMOKE_DEMO_BUILT":
        reasons.append("smoke demo failed")
    if reproduction.get("verdict") != "TWOTANK_MAIN_RESULT_REPRODUCED":
        reasons.append("TwoTank reproduction failed")
    if readme_check.get("verdict") != "README_PUBLIC_LANDING_SYNCED":
        reasons.append("README public landing check failed")
    if custom_run.returncode != 0:
        reasons.append(f"custom model example failed: {custom_run.stderr}")
    if builtin_compare.get("verdict") != "MODEL_COMPARISON_BUILT":
        reasons.append("built-in model comparison failed")
    if custom_compare.get("verdict") != "MODEL_COMPARISON_BUILT":
        reasons.append("custom model comparison failed")
    if claim_scan["violations"]:
        reasons.append("claim-language check failed")
    if hash_mismatches:
        reasons.append(f"prior evidence artifact hash mismatches: {hash_mismatches}")
    if missing:
        reasons.append(f"missing required public files: {missing}")
    verdict = "PUBLIC_BENCHMARK_PACKAGE_ACCEPTED" if not reasons else "PUBLIC_BENCHMARK_PACKAGE_REJECTED"
    manifest = build_public_package_manifest(config, verdict)
    result = {
        "verdict": verdict,
        "preconditions": pre.get("verdict"),
        "smoke_demo": smoke.get("verdict"),
        "twotank_reproduction": reproduction.get("verdict"),
        "readme_check": readme_check.get("verdict"),
        "custom_model_returncode": custom_run.returncode,
        "builtin_model_comparison": builtin_compare.get("verdict"),
        "custom_model_comparison": custom_compare.get("verdict"),
        "claim_language_violations": claim_scan["violations"],
        "prior_artifact_mutation_detected": bool(hash_mismatches),
        "source_hash_mismatches": hash_mismatches,
        "missing": missing,
        "public_package_manifest": manifest["output"],
        "reasons": reasons,
    }
    _write_json(PUBLIC_RESULTS_ROOT / "package_check.json", result)
    write_public_package_check_report(result, "reports/public_benchmark_package_check.md")
    if verdict != "PUBLIC_BENCHMARK_PACKAGE_ACCEPTED":
        raise RuntimeError(f"public benchmark package rejected: {reasons}")
    return result


def build_public_package_manifest(config: dict[str, Any], verdict: str) -> dict[str, Any]:
    manifest = {
        "package_id": config["package_id"],
        "release_name": "v1.2-public-benchmark-strengthening",
        "release_type": "public_packaging_only",
        "scientific_claim_changed": False,
        "allowed_claim": config["allowed_claim"]["text"],
        "verdict": verdict,
        "public_outputs": config["public_outputs"],
        "source_artifacts": config["source_artifacts"],
        "known_limitations": [
            "Current evidence remains weak-positive and low-coverage only.",
            "TwoTank is stronger than CSTR.",
            "CSTR effect is positive but weak.",
            "Custom model outputs are local-only and not benchmark evidence.",
            "Expansion remains blocked.",
        ],
        "output": config["public_outputs"]["public_package_manifest"],
    }
    _write_json(config["public_outputs"]["public_package_manifest"], manifest)
    write_public_release_note(manifest, config["public_outputs"]["public_release_note"])
    return manifest


def write_public_release_note(manifest: dict[str, Any], output: str | Path) -> None:
    text = f"""# Release Note: v1.2 Public Benchmark Strengthening

## What changed

Public README landing, smoke demo, main TwoTank reproduction command, README figure, benchmark task page, failure gallery, custom model template, CI workflow, and public package checker.

## What did not change

The scientific claim did not change. Expansion remains blocked.

## Current allowed claim

{manifest["allowed_claim"]}

## Public usability outputs

{_markdown_table(pd.DataFrame([{"artifact": key, "path": value} for key, value in manifest["public_outputs"].items()]), ["artifact", "path"])}

## Claim boundaries

Do not treat this public packaging milestone as new scientific evidence.

## Verdict

{manifest["verdict"]}
"""
    target = Path(output)
    _ensure_dir(target.parent)
    target.write_text(text, encoding="utf-8")


def write_public_package_check_report(result: dict[str, Any], output: str | Path) -> None:
    text = f"""# Public Benchmark Package Check

## Preconditions

{result["preconditions"]}

## Smoke demo

{result["smoke_demo"]}

## Main TwoTank reproduction

{result["twotank_reproduction"]}

## README check

{result["readme_check"]}

## Custom model example

Return code: {result["custom_model_returncode"]}

## Model comparison

Built-in: {result["builtin_model_comparison"]}

Custom: {result["custom_model_comparison"]}

## Claim-language check

Violations: {result["claim_language_violations"] or "none"}

## Prior artifact mutation detected

{result["prior_artifact_mutation_detected"]}

## Missing files

{result["missing"] or "none"}

## Verdict

{result["verdict"]}
"""
    target = Path(output)
    _ensure_dir(target.parent)
    target.write_text(text, encoding="utf-8")
