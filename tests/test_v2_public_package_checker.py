from __future__ import annotations

import json
from pathlib import Path

import pytest

from scs.experiments.v2_public_hardening import check_public_benchmark_package


def test_v2_public_package_checker_accepts_and_runs_benchmarks() -> None:
    result = check_public_benchmark_package(
        "configs/v2/v2_public_benchmark_hardening.yaml",
        "results/v2_public_benchmark_hardening/package_manifest.json",
    )
    assert result["verdict"] == "V2_PUBLIC_BENCHMARK_PACKAGE_ACCEPTED"
    assert result["custom_model_benchmark_executed"] is True
    assert result["builtin_benchmark_executed"] is True
    assert result["claim_language_hits"] == []


def test_v2_public_package_checker_rejects_method_claim(tmp_path: Path) -> None:
    check_path = Path("results/v2_public_benchmark_hardening/package_check.json")
    report_path = Path("reports/v2_public_benchmark_package_check.md")
    original_check = check_path.read_text(encoding="utf-8") if check_path.exists() else None
    original_report = report_path.read_text(encoding="utf-8") if report_path.exists() else None
    manifest = json.loads(Path("results/v2_public_benchmark_hardening/package_manifest.json").read_text())
    manifest["method_claim_supported"] = True
    bad_manifest = tmp_path / "manifest.json"
    bad_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    try:
        with pytest.raises(RuntimeError):
            check_public_benchmark_package("configs/v2/v2_public_benchmark_hardening.yaml", bad_manifest)
    finally:
        if original_check is not None:
            check_path.write_text(original_check, encoding="utf-8")
        if original_report is not None:
            report_path.write_text(original_report, encoding="utf-8")
