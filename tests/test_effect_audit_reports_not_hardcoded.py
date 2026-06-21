from __future__ import annotations

from pathlib import Path

import pandas as pd

from scs.experiments.effect_audit import write_effect_size_report


def test_effect_size_report_uses_dataframe_values(tmp_path: Path) -> None:
    config = {
        "minimum_absolute_far_reduction": 0.05,
        "minimum_relative_far_reduction": 0.10,
        "confidence_level": 0.95,
        "bootstrap_iterations": 10,
    }
    summary = {"verdict": "WEAK_TWO_SYSTEM_EFFECT", "known_limitations": ["fixture"]}
    effect = pd.DataFrame(
        [
            {
                "system_id": "two_tank",
                "coverage": 0.05,
                "absolute_margin": 0.123456,
                "relative_margin": 0.2,
                "bootstrap_ci_low": 0.1,
                "bootstrap_ci_high": 0.2,
                "seed_ci_low": 0.1,
                "seed_ci_high": 0.2,
                "seed_win_rate": 1.0,
                "meets_absolute_threshold": True,
                "meets_relative_threshold": True,
                "verdict": "PRACTICALLY_MEANINGFUL",
            },
            {
                "system_id": "cstr",
                "coverage": 0.05,
                "absolute_margin": 0.01,
                "relative_margin": 0.02,
                "bootstrap_ci_low": 0.0,
                "bootstrap_ci_high": 0.02,
                "seed_ci_low": 0.0,
                "seed_ci_high": 0.02,
                "seed_win_rate": 0.6,
                "meets_absolute_threshold": False,
                "meets_relative_threshold": False,
                "verdict": "POSITIVE_BUT_WEAK",
            },
        ]
    )
    write_effect_size_report(config, summary, effect, tmp_path / "report.md")
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "0.123456" in text
    effect.loc[0, "absolute_margin"] = 0.654321
    write_effect_size_report(config, summary, effect, tmp_path / "report_b.md")
    assert "0.654321" in (tmp_path / "report_b.md").read_text(encoding="utf-8")
