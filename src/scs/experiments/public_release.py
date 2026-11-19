from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PUBLIC_RELEASE_READY = "PUBLIC_RELEASE_READY"
PUBLIC_RELEASE_BLOCKED = "PUBLIC_RELEASE_BLOCKED"
FRESH_CLONE_REPRO_PASSED = "FRESH_CLONE_REPRO_PASSED"
FRESH_CLONE_REPRO_FAILED = "FRESH_CLONE_REPRO_FAILED"
PUBLIC_RELEASE_PACKAGE_ACCEPTED = "PUBLIC_RELEASE_PACKAGE_ACCEPTED"
PUBLIC_RELEASE_PACKAGE_REJECTED = "PUBLIC_RELEASE_PACKAGE_REJECTED"

README_REQUIRED_SECTIONS = [
    "## What this is",
    "## Quickstart",
    "## Run the benchmark/demo",
    "## Plug in your own model",
    "## Current evidence",
    "## What is not claimed",
    "## Repository structure",
    "## Reproducibility",
    "## Citation",
    "## License",
]

QUICKSTART_COMMANDS = [
    "python -m venv .venv",
    "source .venv/bin/activate",
    'pip install -e ".[dev]"',
    "pytest -q",
    "python scripts/run_smoke.py",
    "python scripts/run_current_status_demo.py",
    "python examples/custom_model_example.py --output results/custom_model_example",
]

FRESH_CLONE_COMMANDS = [
    ["python", "-m", "venv", ".venv"],
    [".venv/bin/python", "-m", "pip", "install", "-e", ".[dev]"],
    [".venv/bin/python", "-m", "pytest", "-q"],
    [".venv/bin/python", "scripts/run_smoke.py"],
    [".venv/bin/python", "scripts/run_current_status_demo.py"],
    [
        ".venv/bin/python",
        "examples/custom_model_example.py",
        "--output",
        "results/custom_model_example",
    ],
]

REQUIRED_PUBLIC_FILES = [
    "LICENSE",
    "README.md",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "docs/benchmark_card.md",
    "docs/reproducibility_card.md",
    "docs/custom_model_adapter.md",
    "docs/public_claims.md",
    "examples/custom_model_example.py",
    "scripts/run_smoke.py",
    "scripts/run_current_status_demo.py",
]

SCAN_ROOTS = [
    "README.md",
    "docs",
    "configs",
    "scripts",
    "src",
    "examples",
    "reports",
    "results",
]
CLAIM_SCAN_ROOTS = [
    "README.md",
    "docs",
    "reports/public_release_exposure_decision.md",
    "reports/release_note_v1_1_public_benchmark.md",
]

PRIVATE_PATTERNS = [
    r"/home/ismayil",
    r"/home/",
    r"C:\\Users",
    r"Agtrup",
    r"Kolding",
    r"\bhistorian\b",
    r"\bcustomer\b",
    r"\bclient\b",
    r"\bprivate\b",
    r"\bsecret\b",
    r"\btoken\b",
    r"\bpassword\b",
    r"\bapi_key\b",
    r"OPENAI_API_KEY",
    r"AWS_SECRET",
    r"DATABASE_URL",
]

FORBIDDEN_CLAIM_PATTERNS = [
    "safety-certified",
    "safety certification",
    "trusted simulator",
    "trustworthy simulator",
    "validated digital twin",
    "production-ready",
    "industrial AI breakthrough",
    "guaranteed reliable",
    "general simulator reliability",
    "works generally",
    "robust calibrated refusal",
    "robust calibrated-refusal",
]

RELEASE_OUTPUT_PATHS = [
    "results/public_release_audit",
    "results/fresh_clone_check",
    "results/public_release_package_check",
    "reports/public_release_readiness_audit.md",
    "reports/fresh_clone_reproduction_check.md",
    "reports/public_release_package_check.md",
    "reports/public_release_exposure_decision.md",
    "reports/public_release_artifact_cleanup.md",
    "reports/release_note_v1_1_public_benchmark.md",
]

LARGE_FILE_BYTES = 20_000_000
LARGE_FILE_JUSTIFICATIONS = {
    "results/v2_scientific_strengthening/frozen_protocol/v2_risk_coverage.csv": (
        "kept as frozen v2 diagnostic evidence used by the public event-risk figure and package checks"
    ),
    "results/v2_scientific_strengthening/frozen_protocol/v2_scenario_scores.csv": (
        "kept as frozen v2 diagnostic evidence used by the public failure gallery and package checks"
    ),
}
CURATED_IGNORED_RELEASE_GLOBS = [
    "results/current_status/**/*.json",
    "results/current_status/**/*.csv",
    "results/calibrated_two_tank/calibrated_judge_summary.json",
    "results/calibrated_two_tank/calibrated_risk_coverage.csv",
    "results/calibrated_two_tank/low_coverage_summary.csv",
    "results/calibrated_two_tank/test_table.csv",
    "results/calibrated_two_tank/data/*",
    "results/calibrated_cstr/calibrated_judge_summary.json",
    "results/calibrated_cstr/calibrated_risk_coverage.csv",
    "results/calibrated_cstr/low_coverage_summary.csv",
    "results/calibrated_cstr/test_table.csv",
    "results/calibrated_cstr/data/*",
    "results/calibrated_seed_sweep_two_tank/seed_sweep_calibrated_summary.json",
    "results/calibrated_seed_sweep_two_tank/calibrated_risk_coverage_all.csv",
    "results/calibrated_seed_sweep_cstr/seed_sweep_calibrated_summary.json",
    "results/calibrated_seed_sweep_cstr/calibrated_risk_coverage_all.csv",
    "results/calibrated_stress_two_tank/stress_summary.json",
    "results/calibrated_stress_cstr/stress_summary.json",
    "results/cstr_sanity/cstr_label_checks.json",
    "results/effect_size_audit/effect_size/*",
    "results/effect_size_audit/event_risk/*",
    "results/effect_size_audit/false_accept_forensics/*",
    "results/cstr_weakness_audit/repair_signal/repair_signal_metrics.json",
    "results/repair_signal_semantics_audit/repair_validation/repair_validation_summary.json",
    "results/repair_signal_semantics_audit/repair_vs_invariant/repair_vs_invariant_summary.json",
    "results/benchmark_usability/package_check.json",
    "results/benchmark_usability/preconditions/*",
    "results/benchmark_usability/release/*",
    "results/public_benchmark_v1_2/*.json",
    "results/public_benchmark_v1_2/preconditions/*",
    "results/technical_note_package/**/*.json",
    "results/technical_note_package/**/*.csv",
    "results/smoke_two_tank/model_metrics.csv",
]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    _ensure_dir(target.parent)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    target = Path(path)
    _ensure_dir(target.parent)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _git(args: list[str], cwd: str | Path = ".") -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()
    except subprocess.CalledProcessError as exc:
        return exc.output.strip()


def _git_list(args: list[str], cwd: str | Path = ".") -> list[str]:
    output = _git(args, cwd=cwd)
    return [line for line in output.splitlines() if line.strip()]


def _text_files_under(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    if not root.exists():
        return []
    if Path(".git").exists() and not root.is_absolute():
        root_text = root.as_posix().rstrip("/")
        release_files = []
        for name in _release_file_list():
            path = Path(name)
            path_text = path.as_posix()
            if path_text == root_text or path_text.startswith(f"{root_text}/"):
                release_files.append(path)
        candidates = release_files
    else:
        candidates = [path for path in root.rglob("*") if path.is_file()]
    files: list[Path] = []
    for path in candidates:
        if not path.is_file() or path.stat().st_size > LARGE_FILE_BYTES:
            continue
        if any(part in {".git", ".venv", "__pycache__", ".pytest_cache"} for part in path.parts):
            continue
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".pyc", ".pkl", ".joblib"}:
            continue
        files.append(path)
    return files


def _is_release_output(path: Path) -> bool:
    text = path.as_posix()
    return any(text == item or text.startswith(f"{item}/") for item in RELEASE_OUTPUT_PATHS)


def _line_is_non_claim_context(line: str, active_context: str) -> bool:
    lower = line.lower().strip()
    if active_context == "non_claim":
        return True
    markers = [
        "not claimed",
        "non-claim",
        "non-claims",
        "claim boundaries",
        "forbidden",
        "non-intended",
        "known limitations",
        "what is not claimed",
        "what this does not",
        "what is not supported",
        "risk phrases",
    ]
    negations = [
        "does not claim",
        "does not support",
        "does not provide",
        "does not prove",
        "do not claim",
        "do not use",
        "is not a",
        "is not an",
        "is not ",
        "not a ",
        "not a safety",
        "not a product",
        "not evidence",
        "not safety certification",
        "not a claim",
    ]
    return any(marker in lower for marker in markers) or any(negation in lower for negation in negations)


def scan_claim_language(paths: list[str | Path] | None = None) -> list[dict[str, Any]]:
    roots = [Path(p) for p in (paths or CLAIM_SCAN_ROOTS)]
    hits: list[dict[str, Any]] = []
    for root in roots:
        for path in _text_files_under(root):
            if _is_release_output(path):
                continue
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            context = "body"
            for line_no, line in enumerate(lines, start=1):
                stripped = line.strip()
                lower = stripped.lower()
                if lower.startswith("#"):
                    context = "non_claim" if _line_is_non_claim_context(stripped, "body") else "body"
                marker_line = stripped.strip(" *")
                if marker_line.endswith(":") and _line_is_non_claim_context(marker_line, "body"):
                    context = "non_claim"
                    continue
                if _line_is_non_claim_context(stripped, context):
                    continue
                for pattern in FORBIDDEN_CLAIM_PATTERNS:
                    if pattern in lower:
                        hits.append(
                            {
                                "path": path.as_posix(),
                                "line": line_no,
                                "pattern": pattern,
                                "status": "unresolved",
                                "text": stripped[:240],
                            }
                        )
    return hits


def _private_hit_status(path: Path, line: str, pattern: str) -> tuple[str, str]:
    if _is_release_output(path):
        return "allowlisted", "release audit output discusses release risks"
    if path.as_posix() in {
        "src/scs/experiments/public_release.py",
        "scripts/check_public_release_ready.py",
        "scripts/check_public_release_package.py",
    }:
        return "allowlisted", "release checker contains required scan patterns"
    if path.as_posix() in {"docs/public_claims.md", "CONTRIBUTING.md"}:
        return "allowlisted", "public documentation describes prohibited disclosure classes"
    if pattern == r"\btoken\b" and path.suffix == ".py":
        return "allowlisted", "generic source-code variable or scanner term, not a credential"
    lower = line.lower()
    if any(token in lower for token in ["private data", "private/local", "secret-like", "no secrets"]):
        return "allowlisted", "public release safety wording"
    return "unresolved", ""


def scan_private_patterns(paths: list[str | Path] | None = None) -> list[dict[str, Any]]:
    roots = [Path(p) for p in (paths or SCAN_ROOTS)]
    compiled = [(pattern, re.compile(pattern, flags=re.IGNORECASE)) for pattern in PRIVATE_PATTERNS]
    hits: list[dict[str, Any]] = []
    for root in roots:
        for path in _text_files_under(root):
            if path.as_posix().startswith("results/public_release_audit/"):
                continue
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            for line_no, line in enumerate(lines, start=1):
                for pattern, regex in compiled:
                    if regex.search(line):
                        status, reason = _private_hit_status(path, line, pattern)
                        hits.append(
                            {
                                "path": path.as_posix(),
                                "line": line_no,
                                "pattern": pattern,
                                "status": status,
                                "allowlist_reason": reason,
                                "text": line.strip()[:240],
                            }
                        )
    return hits


def _release_file_list() -> list[str]:
    tracked = set(_git_list(["ls-files"]))
    untracked = set(_git_list(["ls-files", "--others", "--exclude-standard"]))
    curated: set[str] = set()
    for pattern in CURATED_IGNORED_RELEASE_GLOBS:
        for path in Path(".").glob(pattern):
            if path.is_file():
                curated.add(path.as_posix())
    return sorted(tracked | untracked | curated)


def scan_large_files(limit: int = LARGE_FILE_BYTES) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in _release_file_list():
        path = Path(name)
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size <= limit:
            continue
        justification = LARGE_FILE_JUSTIFICATIONS.get(path.as_posix(), "")
        rows.append(
            {
                "path": path.as_posix(),
                "bytes": size,
                "status": "justified" if justification else "unresolved",
                "justification": justification,
            }
        )
    return rows


def repo_size_bytes() -> int:
    total = 0
    for name in _release_file_list():
        path = Path(name)
        if path.is_file():
            total += path.stat().st_size
    return total


def validate_readme() -> list[str]:
    reasons: list[str] = []
    readme = Path("README.md")
    if not readme.exists():
        return ["README.md missing"]
    text = readme.read_text(encoding="utf-8")
    required_opening = (
        "# Selective Counterfactual Simulation Benchmark\n\n"
        "A benchmark for testing whether learned dynamical simulators know when to refuse counterfactual predictions.\n\n"
        "Plug in a simulator, run OOD/intervention scenarios, and compare false-accept rate versus coverage.\n\n"
        "**Current status:** this is a benchmark prototype with narrow synthetic evidence. "
        "It is not a safety tool, product-ready digital twin, or claim of general simulator reliability."
    )
    if not text.startswith(required_opening):
        reasons.append("README does not start with required public release opening")
    for section in README_REQUIRED_SECTIONS:
        if section not in text:
            reasons.append(f"README missing section: {section}")
    for command in QUICKSTART_COMMANDS:
        if command not in text:
            reasons.append(f"README missing quickstart command: {command}")
    for non_claim in [
        "This is not safety certification.",
        "This is not a product-ready digital twin.",
        "This is not a claim of general simulator reliability.",
        "This is not an autonomous control system.",
        "This is not evidence that calibrated refusal works generally.",
    ]:
        if non_claim not in text:
            reasons.append(f"README missing non-claim: {non_claim}")
    return reasons


def check_public_release_ready(output: str | Path) -> dict[str, Any]:
    out_dir = Path(output)
    _ensure_dir(out_dir)
    private_hits = scan_private_patterns()
    claim_hits = scan_claim_language()
    large_files = scan_large_files()
    missing = [path for path in REQUIRED_PUBLIC_FILES if not Path(path).exists() or Path(path).stat().st_size == 0]
    readme_reasons = validate_readme()
    unresolved_private = [hit for hit in private_hits if hit["status"] != "allowlisted"]
    unresolved_large = [row for row in large_files if row["status"] != "justified"]
    reasons: list[str] = []
    if missing:
        reasons.append(f"missing required public files: {missing}")
    reasons.extend(readme_reasons)
    if unresolved_private:
        reasons.append(f"unresolved private/local pattern hits: {len(unresolved_private)}")
    if claim_hits:
        reasons.append(f"forbidden claim language hits: {len(claim_hits)}")
    if unresolved_large:
        reasons.append(f"unresolved large files: {len(unresolved_large)}")
    verdict = PUBLIC_RELEASE_READY if not reasons else PUBLIC_RELEASE_BLOCKED
    _write_csv(
        out_dir / "private_pattern_hits.csv",
        private_hits,
        ["path", "line", "pattern", "status", "allowlist_reason", "text"],
    )
    _write_csv(
        out_dir / "large_files.csv",
        large_files,
        ["path", "bytes", "status", "justification"],
    )
    _write_csv(
        out_dir / "claim_language_hits.csv",
        claim_hits,
        ["path", "line", "pattern", "status", "text"],
    )
    result = {
        "verdict": verdict,
        "current_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "latest_commit": _git(["rev-parse", "HEAD"]),
        "working_tree_status": _git(["status", "--short"]),
        "repo_size_bytes_tracked_and_nonignored": repo_size_bytes(),
        "required_files_missing": missing,
        "private_pattern_hits": len(private_hits),
        "private_pattern_unresolved": len(unresolved_private),
        "large_files": len(large_files),
        "large_files_unresolved": len(unresolved_large),
        "claim_language_hits": len(claim_hits),
        "readme_errors": readme_reasons,
        "reasons": reasons,
    }
    _write_json(out_dir / "public_release_check.json", result)
    write_public_release_readiness_report(result, private_hits, large_files, claim_hits)
    write_artifact_cleanup_report(large_files)
    return result


def write_public_release_readiness_report(
    result: dict[str, Any],
    private_hits: list[dict[str, Any]],
    large_files: list[dict[str, Any]],
    claim_hits: list[dict[str, Any]],
) -> None:
    lines = [
        "# Public Release Readiness Audit",
        "",
        "## Verdict",
        "",
        result["verdict"],
        "",
        "## Git state",
        "",
        f"- branch: {result['current_branch']}",
        f"- latest commit: {result['latest_commit']}",
        f"- working tree status lines: {len(result['working_tree_status'].splitlines()) if result['working_tree_status'] else 0}",
        "",
        "## Required files",
        "",
        f"Missing: {result['required_files_missing'] or ['none']}",
        "",
        "## Private/local pattern scan",
        "",
        f"- hits: {len(private_hits)}",
        f"- unresolved: {result['private_pattern_unresolved']}",
        "",
        "## Large file scan",
        "",
        f"- files over 20 MB: {len(large_files)}",
        f"- unresolved: {result['large_files_unresolved']}",
        "",
        "## Claim language scan",
        "",
        f"- unresolved forbidden claim hits: {len(claim_hits)}",
        "",
        "## Reasons",
        "",
        str(result["reasons"] or ["none"]),
    ]
    Path("reports/public_release_readiness_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_artifact_cleanup_report(large_files: list[dict[str, Any]]) -> None:
    repo_size = repo_size_bytes()
    gitignore = Path(".gitignore").read_text(encoding="utf-8") if Path(".gitignore").exists() else ""
    required_ignores = [".env", ".env.*", "results/tmp/", "results/local/", "*.log", ".ipynb_checkpoints/"]
    missing_ignores = [item for item in required_ignores if item not in gitignore]
    unresolved = [row for row in large_files if row["status"] != "justified"]
    verdict = "PUBLIC_ARTIFACTS_ACCEPTED" if not unresolved and not missing_ignores else "PUBLIC_ARTIFACTS_BLOCKED"
    lines = [
        "# Public Release Artifact Cleanup",
        "",
        "## Repo size before",
        "",
        f"{repo_size} bytes tracked/nonignored.",
        "",
        "## Large files found",
        "",
        json.dumps(large_files, indent=2),
        "",
        "## Files kept intentionally",
        "",
        json.dumps([row for row in large_files if row["status"] == "justified"], indent=2),
        "",
        "## Files removed",
        "",
        "none",
        "",
        "## Gitignore changes",
        "",
        f"Missing required ignore patterns: {missing_ignores or ['none']}",
        "",
        "## Repo size after",
        "",
        f"{repo_size} bytes tracked/nonignored.",
        "",
        "## Verdict",
        "",
        verdict,
    ]
    Path("reports/public_release_artifact_cleanup.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_release_tree(source: Path, destination: Path) -> None:
    _ensure_dir(destination)
    names = _release_file_list()
    for name in names:
        src = source / name
        if not src.exists() or not src.is_file():
            continue
        dst = destination / name
        _ensure_dir(dst.parent)
        shutil.copy2(src, dst)


@dataclass
class CommandResult:
    command: str
    returncode: int
    duration_seconds: float


def _run_command(command: list[str], cwd: Path, log_handle: Any, timeout: int = 600) -> CommandResult:
    import time

    started = time.monotonic()
    printable = " ".join(command)
    log_handle.write(f"\n$ {printable}\n")
    log_handle.flush()
    proc = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    elapsed = time.monotonic() - started
    log_handle.write(proc.stdout)
    log_handle.write(f"\n[exit {proc.returncode} after {elapsed:.2f}s]\n")
    log_handle.flush()
    return CommandResult(printable, proc.returncode, elapsed)


def check_fresh_clone_repro(repo_path: str | Path, output: str | Path) -> dict[str, Any]:
    source = Path(repo_path).resolve()
    out_dir = Path(output)
    _ensure_dir(out_dir)
    command_log = out_dir / "command_log.txt"
    command_results: list[CommandResult] = []
    with tempfile.TemporaryDirectory(prefix="scs_fresh_clone_") as tmp:
        clone_dir = Path(tmp) / "selective-counterfactual-simulation"
        copy_release_tree(source, clone_dir)
        with command_log.open("w", encoding="utf-8") as log_handle:
            log_handle.write("Fresh clone reproduction check\n")
            log_handle.write(f"source_commit={_git(['rev-parse', 'HEAD'], cwd=source)}\n")
            for command in FRESH_CLONE_COMMANDS:
                result = _run_command(command, clone_dir, log_handle)
                command_results.append(result)
                if result.returncode != 0:
                    break
    failed = [result for result in command_results if result.returncode != 0]
    verdict = FRESH_CLONE_REPRO_PASSED if not failed and len(command_results) == len(FRESH_CLONE_COMMANDS) else FRESH_CLONE_REPRO_FAILED
    status = {
        "verdict": verdict,
        "source_commit": _git(["rev-parse", "HEAD"], cwd=source),
        "commands": [
            {
                "command": result.command,
                "returncode": result.returncode,
                "duration_seconds": round(result.duration_seconds, 3),
            }
            for result in command_results
        ],
        "log": str(command_log),
    }
    _write_json(out_dir / "command_status.json", status)
    write_fresh_clone_report(status)
    return status


def write_fresh_clone_report(status: dict[str, Any]) -> None:
    lines = [
        "# Fresh Clone Reproduction Check",
        "",
        "## Verdict",
        "",
        status["verdict"],
        "",
        "## Commands",
        "",
    ]
    for row in status["commands"]:
        lines.append(f"- `{row['command']}`: exit {row['returncode']} ({row['duration_seconds']}s)")
    lines.extend(["", "## Log", "", status["log"]])
    Path("reports/fresh_clone_reproduction_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def check_public_release_package(output: str | Path) -> dict[str, Any]:
    out_dir = Path(output)
    _ensure_dir(out_dir)
    audit = check_public_release_ready("results/public_release_audit")
    fresh = check_fresh_clone_repro(Path.cwd(), "results/fresh_clone_check")
    readme_errors = validate_readme()
    docs_missing = [path for path in REQUIRED_PUBLIC_FILES if not Path(path).exists() or Path(path).stat().st_size == 0]
    reasons: list[str] = []
    if audit["verdict"] != PUBLIC_RELEASE_READY:
        reasons.append("public release audit did not pass")
    if fresh["verdict"] != FRESH_CLONE_REPRO_PASSED:
        reasons.append("fresh clone reproduction did not pass")
    if readme_errors:
        reasons.extend(readme_errors)
    if docs_missing:
        reasons.append(f"missing docs/files: {docs_missing}")
    verdict = PUBLIC_RELEASE_PACKAGE_ACCEPTED if not reasons else PUBLIC_RELEASE_PACKAGE_REJECTED
    result = {
        "verdict": verdict,
        "public_release_audit": audit["verdict"],
        "fresh_clone_reproduction": fresh["verdict"],
        "private_pattern_unresolved": audit["private_pattern_unresolved"],
        "claim_language_hits": audit["claim_language_hits"],
        "large_files_unresolved": audit["large_files_unresolved"],
        "readme_errors": readme_errors,
        "reasons": reasons,
    }
    _write_json(out_dir / "package_check.json", result)
    write_public_release_package_report(result)
    return result


def write_public_release_package_report(result: dict[str, Any]) -> None:
    lines = [
        "# Public Release Package Check",
        "",
        "## Verdict",
        "",
        result["verdict"],
        "",
        "## Subchecks",
        "",
        f"- public release audit: {result['public_release_audit']}",
        f"- fresh clone reproduction: {result['fresh_clone_reproduction']}",
        f"- unresolved private/local hits: {result['private_pattern_unresolved']}",
        f"- forbidden claim hits: {result['claim_language_hits']}",
        f"- unresolved large files: {result['large_files_unresolved']}",
        "",
        "## Reasons",
        "",
        str(result["reasons"] or ["none"]),
    ]
    Path("reports/public_release_package_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
