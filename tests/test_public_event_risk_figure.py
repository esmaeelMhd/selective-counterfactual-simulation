from __future__ import annotations

import json
from pathlib import Path

import yaml

from scs.experiments.v2_public_hardening import build_public_event_risk_figure


def test_public_event_risk_figure_values_and_caption(tmp_path: Path) -> None:
    out = tmp_path / "event.png"
    manifest = build_public_event_risk_figure("configs/v2/v2_public_benchmark_hardening.yaml", out)
    assert out.exists()
    assert out.stat().st_size > 0
    assert manifest["event_target_mean_margin"] < 0.0
    assert "not a method-success claim" in manifest["caption"]
    report = Path("reports/v2_public_event_risk_figure.md").read_text(encoding="utf-8")
    assert "Event target mean margin" in report


def test_public_event_risk_figure_uses_source_values(tmp_path: Path) -> None:
    report_path = Path("reports/v2_public_event_risk_figure.md")
    manifest_path = Path("results/v2_public_benchmark_hardening/figures/event_risk_figure_manifest.json")
    original_report = report_path.read_text(encoding="utf-8") if report_path.exists() else None
    original_manifest = manifest_path.read_text(encoding="utf-8") if manifest_path.exists() else None
    config = yaml.safe_load(Path("configs/v2/v2_public_benchmark_hardening.yaml").read_text(encoding="utf-8"))
    stats = json.loads(Path(config["source_artifacts"]["comparator_stats"]).read_text(encoding="utf-8"))
    stats["rmse_target_result"]["mean_margin"] = 0.12345
    stats["event_target_result"]["mean_margin"] = -0.54321
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(json.dumps(stats), encoding="utf-8")
    config["source_artifacts"]["comparator_stats"] = str(stats_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    try:
        manifest = build_public_event_risk_figure(config_path, tmp_path / "changed.png")
    finally:
        if original_report is not None:
            report_path.write_text(original_report, encoding="utf-8")
        if original_manifest is not None:
            manifest_path.write_text(original_manifest, encoding="utf-8")
    assert manifest["rmse_target_mean_margin"] == 0.12345
    assert manifest["event_target_mean_margin"] == -0.54321
