from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scs.reports.failure_analysis import SIGNALS, _safe_auc, analyze_signal_error_correlation


def test_signal_error_correlation_outputs_real_statistics(tmp_path: Path, failure_table_path: Path) -> None:
    summary = analyze_signal_error_correlation(failure_table_path, tmp_path)
    result = pd.read_csv(tmp_path / "signal_error_correlation.csv")
    assert summary["verdict"] in {"NO_SIGNAL", "WEAK_SIGNALS", "USEFUL_SIGNALS_FOUND"}
    assert "oracle_error_rank_score" not in SIGNALS
    assert result["target"].isin(["bad_rmse_label"]).any()
    assert result["auroc_for_bad_rmse"].dropna().between(0.0, 1.0).all()
    assert not result[result["signal"] == "support_distance"].empty


def test_safe_auc_returns_nan_for_degenerate_labels() -> None:
    assert np.isnan(_safe_auc(pd.Series([0.1, 0.2, 0.3]), pd.Series([1, 1, 1]), "auroc"))
