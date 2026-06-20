from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scs.reports.claim_audit import write_claim_audit


def _write_fixture(path: Path, combined_far: float, best_far: float, oracle_far: float = 0.0) -> None:
    path.mkdir(parents=True, exist_ok=True)
    rows = []
    for split in ["ood_action_magnitude", "ood_inflow_spike"]:
        for model_id in ["linear_narx"]:
            for coverage in [0.5]:
                judge_scores = {
                    "combined_linear": combined_far,
                    "support_only": best_far,
                    "uncertainty_only": best_far + 0.1,
                    "disagreement_only": best_far + 0.2,
                    "invariant_only": best_far + 0.3,
                    "repair_only": best_far + 0.4,
                    "random_baseline": best_far + 0.5,
                    "oracle_error_rank": oracle_far,
                }
                for judge_id, far in judge_scores.items():
                    rows.append(
                        {
                            "system_id": "two_tank",
                            "model_id": model_id,
                            "split": split,
                            "judge_id": judge_id,
                            "coverage": coverage,
                            "false_accept_rate": far,
                            "accepted_count": 2,
                            "false_accept_count": int(far > 0),
                            "mean_error_accepted": far,
                            "mean_error_rejected": 0.0,
                            "threshold": 0.15,
                        }
                    )
    pd.DataFrame(rows).to_csv(path / "risk_coverage.csv", index=False)
    pd.DataFrame({"x": [1]}).to_csv(path / "scenario_scores.csv", index=False)
    pd.DataFrame({"x": [1]}).to_csv(path / "model_metrics.csv", index=False)
    (path / "summary.json").write_text("{}", encoding="utf-8")


def test_claim_audit_supported_and_oracle_excluded(tmp_path) -> None:
    _write_fixture(tmp_path, combined_far=0.1, best_far=0.3, oracle_far=0.0)
    summary = write_claim_audit(tmp_path, tmp_path / "report.md")
    assert summary["verdict"] == "SUPPORTED"
    audit = pd.read_csv(tmp_path / "claim_audit.csv")
    assert set(audit["best_simple_judge"]) == {"support_only"}


def test_claim_audit_not_supported(tmp_path) -> None:
    _write_fixture(tmp_path, combined_far=0.5, best_far=0.2, oracle_far=0.0)
    summary = write_claim_audit(tmp_path, tmp_path / "report.md")
    assert summary["verdict"] == "NOT_SUPPORTED"


def test_claim_audit_tie_is_mixed_not_win(tmp_path) -> None:
    _write_fixture(tmp_path, combined_far=0.2, best_far=0.2, oracle_far=0.0)
    summary = write_claim_audit(tmp_path, tmp_path / "report.md")
    audit = pd.read_csv(tmp_path / "claim_audit.csv")
    assert summary["verdict"] == "NOT_SUPPORTED"
    assert audit["combined_ties"].all()
    assert not audit["combined_wins"].any()

