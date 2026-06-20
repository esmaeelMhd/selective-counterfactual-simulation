from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scs.reports.multi_system import make_multi_system_report


def _write_result_fixture(path: Path, system_id: str, combined_far: float, support_far: float) -> None:
    path.mkdir(parents=True, exist_ok=True)
    rows = []
    judge_far = {
        "combined_linear": combined_far,
        "support_only": support_far,
        "uncertainty_only": support_far + 0.1,
        "disagreement_only": support_far + 0.2,
        "invariant_only": support_far + 0.3,
        "repair_only": support_far + 0.4,
        "random_baseline": support_far + 0.5,
        "oracle_error_rank": 0.0,
    }
    for split in ["id_test", "ood_action_magnitude", "ood_combined"]:
        for coverage in [0.5, 1.0]:
            for judge_id, false_accept_rate in judge_far.items():
                rows.append(
                    {
                        "system_id": system_id,
                        "model_id": "linear_narx",
                        "split": split,
                        "judge_id": judge_id,
                        "coverage": coverage,
                        "false_accept_rate": false_accept_rate,
                        "accepted_count": 2,
                        "false_accept_count": int(false_accept_rate > 0),
                        "mean_error_accepted": false_accept_rate,
                        "mean_error_rejected": 0.0,
                        "threshold": 0.15,
                    }
                )
    pd.DataFrame(rows).to_csv(path / "risk_coverage.csv", index=False)
    pd.DataFrame({"x": [1]}).to_csv(path / "scenario_scores.csv", index=False)
    pd.DataFrame({"x": [1]}).to_csv(path / "model_metrics.csv", index=False)
    (path / "summary.json").write_text("{}", encoding="utf-8")


def test_multi_system_report_uses_real_result_dirs_and_gate(tmp_path) -> None:
    two_tank = tmp_path / "two_tank"
    cstr = tmp_path / "cstr"
    _write_result_fixture(two_tank, "two_tank", combined_far=0.3, support_far=0.2)
    _write_result_fixture(cstr, "cstr", combined_far=0.1, support_far=0.3)
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"decision": "KILL_OR_DOWNGRADE_CLAIM"}), encoding="utf-8")

    output = tmp_path / "multi.md"
    report = make_multi_system_report([two_tank, cstr], output, gate_path=gate)

    assert report["overall_claim_status"] == "NOT_SUPPORTED"
    assert output.exists()
    text = output.read_text(encoding="utf-8")
    assert "two_tank" in text
    assert "cstr" in text
    assert "KILL_OR_DOWNGRADE_CLAIM" in text
    assert (tmp_path / "multi.json").exists()
