from __future__ import annotations

from pathlib import Path

from scs.experiments.public_benchmark import build_failure_gallery


def test_failure_gallery_generated_from_actual_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "failure_gallery.md"
    figure_dir = tmp_path / "figures"

    manifest = build_failure_gallery("configs/status/public_benchmark_v1_2.yaml", output, figure_dir)

    assert manifest["verdict"] == "FAILURE_GALLERY_BUILT"
    assert len(manifest["examples"]) >= 4
    assert any(item["system"] == "cstr" and item["false_accept"] for item in manifest["examples"])
    assert any(item["system"] == "cstr" and item["key_signal_values"]["repair_amount"] == 0.0 for item in manifest["examples"])
    assert any(item["key_signal_values"]["invariant_residual"] > 0.05 for item in manifest["examples"])
    assert all("results/calibrated_" in item["source_artifact"] for item in manifest["examples"])
    assert "Key signal values" in output.read_text(encoding="utf-8")
    assert list(figure_dir.glob("*.png"))
