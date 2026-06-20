from __future__ import annotations

import numpy as np
import pandas as pd

SIGNAL_COLUMNS = [
    "support_distance",
    "uncertainty",
    "disagreement",
    "invariant_residual",
    "repair_amount",
]

JUDGE_IDS = [
    "support_only",
    "uncertainty_only",
    "disagreement_only",
    "invariant_only",
    "repair_only",
    "combined_linear",
    "random_baseline",
    "oracle_error_rank",
]


def _normalize(values: pd.Series) -> pd.Series:
    minimum = float(values.min())
    maximum = float(values.max())
    denom = maximum - minimum
    if denom <= 1e-12:
        return pd.Series(np.zeros(len(values)), index=values.index, dtype=float)
    return (values - minimum) / denom


def score_judge(judge_id: str, signals: dict[str, float]) -> float:
    if judge_id == "support_only":
        return float(signals["support_distance"])
    if judge_id == "uncertainty_only":
        return float(signals["uncertainty"])
    if judge_id == "disagreement_only":
        return float(signals["disagreement"])
    if judge_id == "invariant_only":
        return float(signals["invariant_residual"])
    if judge_id == "repair_only":
        return float(signals["repair_amount"])
    if judge_id == "combined_linear":
        return float(sum(signals[column] for column in SIGNAL_COLUMNS))
    if judge_id == "random_baseline":
        return float(signals.get("random_baseline", 0.5))
    if judge_id == "oracle_error_rank":
        return float(signals["error"])
    raise ValueError(f"unknown judge_id: {judge_id}")


def compute_judge_score_frame(
    signal_df: pd.DataFrame,
    judge_ids: list[str],
    seed: int,
) -> pd.DataFrame:
    missing = [column for column in SIGNAL_COLUMNS + ["error"] if column not in signal_df.columns]
    if missing:
        raise ValueError(f"missing signal columns: {missing}")
    scores = pd.DataFrame(index=signal_df.index)
    if "support_only" in judge_ids:
        scores["support_only"] = signal_df["support_distance"].astype(float)
    if "uncertainty_only" in judge_ids:
        scores["uncertainty_only"] = signal_df["uncertainty"].astype(float)
    if "disagreement_only" in judge_ids:
        scores["disagreement_only"] = signal_df["disagreement"].astype(float)
    if "invariant_only" in judge_ids:
        scores["invariant_only"] = signal_df["invariant_residual"].astype(float)
    if "repair_only" in judge_ids:
        scores["repair_only"] = signal_df["repair_amount"].astype(float)
    if "combined_linear" in judge_ids:
        normalized = [_normalize(signal_df[column].astype(float)) for column in SIGNAL_COLUMNS]
        scores["combined_linear"] = sum(normalized) / len(normalized)
    if "random_baseline" in judge_ids:
        rng = np.random.default_rng(seed)
        scores["random_baseline"] = rng.uniform(0.0, 1.0, size=len(signal_df))
    if "oracle_error_rank" in judge_ids:
        scores["oracle_error_rank"] = signal_df["error"].astype(float)

    unknown = sorted(set(judge_ids) - set(scores.columns))
    if unknown:
        raise ValueError(f"unknown judge ids: {unknown}")
    if not np.isfinite(scores.to_numpy(dtype=float)).all():
        raise ValueError("judge scores contain non-finite values")
    return scores[judge_ids]

