from __future__ import annotations

import pandas as pd

from scs.experiments.cstr_weakness import _signal_group, _signal_metrics_for_frame


def test_accepted_good_and_bad_groups_are_defined() -> None:
    frame = pd.DataFrame({"accepted": [True, True, False, False], "bad_rmse_label": [False, True, False, True]})
    assert list(_signal_group(frame)) == ["accepted_good", "accepted_bad", "rejected_good", "rejected_bad"]


def test_signal_separability_metrics_are_computed() -> None:
    frame = pd.DataFrame(
        {
            "signal_group": ["accepted_good", "accepted_good", "accepted_bad", "accepted_bad"],
            "support_distance": [0.1, 0.2, 0.8, 0.9],
        }
    )
    result = _signal_metrics_for_frame(frame, "support_distance")
    assert result["auroc"] == 1.0
    assert result["verdict"] == "SEPARATES"


def test_too_few_samples_case_is_explicit() -> None:
    frame = pd.DataFrame({"signal_group": ["accepted_good", "accepted_bad"], "support_distance": [0.1, 0.9]})
    assert _signal_metrics_for_frame(frame, "support_distance")["verdict"] == "TOO_FEW_SAMPLES"
