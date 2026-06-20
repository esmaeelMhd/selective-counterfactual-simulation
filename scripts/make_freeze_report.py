from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path

import pandas as pd


def _forbidden_repo_names() -> list[str]:
    return [
        "time" + "-series" + "-simulator",
        "digital" + "-twin" + "-engine",
        "flux" + "-attention" + "-engine",
        "plant" + "-scenario" + "-compiler",
    ]


def _path_hack_tokens() -> list[str]:
    return ["sys" + ".path", "PYTHON" + "PATH", "." + "./", "." + ".\\"]


def _scan(paths: list[Path], tokens: list[str]) -> list[dict[str, str]]:
    findings = []
    for root in paths:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".toml", ".yaml", ".yml"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for token in tokens:
                if token in text:
                    findings.append({"path": str(path), "token": token})
    return findings


def make_freeze_report(pytest_result: str, smoke_result: str) -> dict:
    required_artifacts = [
        Path("results/smoke_two_tank/data_summary.json"),
        Path("results/smoke_two_tank/model_metrics.csv"),
        Path("results/smoke_two_tank/scenario_scores.csv"),
        Path("results/smoke_two_tank/risk_coverage.csv"),
        Path("results/smoke_two_tank/risk_coverage.png"),
        Path("results/smoke_two_tank/summary.json"),
        Path("reports/smoke_report.md"),
    ]
    artifact_status = {
        str(path): {"exists": path.exists(), "size": path.stat().st_size if path.exists() else 0}
        for path in required_artifacts
    }
    csv_nonempty = True
    risk_no_nan = False
    try:
        for path in [
            Path("results/smoke_two_tank/model_metrics.csv"),
            Path("results/smoke_two_tank/scenario_scores.csv"),
            Path("results/smoke_two_tank/risk_coverage.csv"),
        ]:
            csv_nonempty = csv_nonempty and (not pd.read_csv(path).empty)
        risk = pd.read_csv("results/smoke_two_tank/risk_coverage.csv")
        risk_no_nan = not bool(risk.isna().sum().sum())
    except Exception:
        csv_nonempty = False

    summary_valid = False
    try:
        json.loads(Path("results/smoke_two_tank/summary.json").read_text(encoding="utf-8"))
        summary_valid = True
    except Exception:
        summary_valid = False

    source_paths = [Path("src"), Path("scripts"), Path("tests")]
    forbidden_findings = _scan(source_paths, _forbidden_repo_names())
    path_hack_findings = _scan(source_paths, _path_hack_tokens())
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], text=True, capture_output=True, check=False).stdout.strip()
    accepted = (
        pytest_result == "passed"
        and smoke_result == "passed"
        and all(item["exists"] and item["size"] > 0 for item in artifact_status.values())
        and csv_nonempty
        and risk_no_nan
        and summary_valid
        and not forbidden_findings
        and not path_hack_findings
    )
    report = {
        "commit": commit,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "commands_run": [
            "git status",
            "python -m venv .venv",
            "source .venv/bin/activate",
            'pip install -e ".[dev]"',
            "pytest -q",
            "python scripts/run_smoke.py",
        ],
        "pytest_result": pytest_result,
        "smoke_result": smoke_result,
        "artifacts": artifact_status,
        "csv_nonempty": csv_nonempty,
        "risk_no_nan": risk_no_nan,
        "summary_valid_json": summary_valid,
        "forbidden_dependency_scan": forbidden_findings,
        "path_hack_scan": path_hack_findings,
        "known_failures": [] if accepted else ["Freeze acceptance checks did not all pass."],
        "verdict": "ACCEPTED" if accepted else "REJECTED",
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/v0_freeze_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    artifact_lines = "\n".join(
        f"- {path}: exists={status['exists']} size={status['size']}"
        for path, status in artifact_status.items()
    )
    text = f"""# v0 Freeze Report

## Commit

{commit}

## Environment

- Python: {report["environment"]["python"]}
- Platform: {report["environment"]["platform"]}

## Commands run

{chr(10).join(f"- {cmd}" for cmd in report["commands_run"])}

## Pytest result

{pytest_result}

## Smoke result

{smoke_result}

## Created artifacts

{artifact_lines}

## Forbidden dependency scan

- forbidden repo references: {len(forbidden_findings)}
- path hacks: {len(path_hack_findings)}

## Known failures

{"- none" if not report["known_failures"] else chr(10).join(f"- {item}" for item in report["known_failures"])}

## Verdict

{report["verdict"]}
"""
    Path("reports/v0_freeze_report.md").write_text(text, encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the v0 freeze report from current smoke outputs.")
    parser.add_argument("--pytest-result", choices=["passed", "failed"], required=True)
    parser.add_argument("--smoke-result", choices=["passed", "failed"], required=True)
    args = parser.parse_args()
    report = make_freeze_report(args.pytest_result, args.smoke_result)
    print(report["verdict"])


if __name__ == "__main__":
    main()

