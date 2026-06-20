from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class CalibratedJudge(Protocol):
    judge_id: str

    def fit(
        self,
        calibration_table: pd.DataFrame,
        signal_columns: list[str],
        error_column: str,
        bad_label_column: str,
    ) -> None:
        ...

    def score(self, scenario_table: pd.DataFrame) -> np.ndarray:
        ...

    def provenance(self) -> dict:
        ...


REQUIRED_PROVENANCE_KEYS = [
    "judge_id",
    "fit_row_count",
    "fit_scenario_count",
    "fit_split_names",
    "signal_columns_used",
    "error_column_used",
    "bad_label_column_used",
    "selected_hyperparameters",
    "selected_signal_if_any",
    "calibration_scenario_id_hash",
    "test_scenario_id_hash",
    "used_test_labels_during_fit",
    "available",
    "unavailable_reason",
]


def scenario_id_hash(ids: pd.Series | list[str]) -> str:
    values = sorted(str(value) for value in list(ids))
    payload = "\n".join(values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _ensure_no_test_rows(table: pd.DataFrame) -> None:
    if "role" in table and table["role"].astype(str).str.contains("test", case=False).any():
        raise ValueError("calibrated judge fit received test role rows")
    if "split" in table and table["split"].astype(str).str.startswith("judge_test").any():
        raise ValueError("calibrated judge fit received judge_test rows")


def _safe_auc(score: pd.Series, label: pd.Series) -> float:
    frame = pd.concat([score, label], axis=1).dropna()
    if len(frame) < 2:
        return float("nan")
    y_score = frame.iloc[:, 0].astype(float)
    y_true = frame.iloc[:, 1].astype(int)
    if y_true.nunique() <= 1 or y_score.nunique() <= 1:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _acceptance_false_accept_rate(errors: np.ndarray, scores: np.ndarray, threshold: float, coverage: float) -> float:
    order = np.argsort(scores, kind="mergesort")
    n_accept = min(max(int(np.ceil(float(coverage) * len(scores))), 1), len(scores))
    accepted_errors = errors[order[:n_accept]]
    return float(np.mean(accepted_errors > threshold))


def _empirical_percentile(calibration_values: np.ndarray, values: np.ndarray) -> np.ndarray:
    sorted_values = np.sort(np.asarray(calibration_values, dtype=float))
    if len(sorted_values) == 0:
        return np.zeros(len(values), dtype=float)
    return np.searchsorted(sorted_values, np.asarray(values, dtype=float), side="right") / len(sorted_values)


@dataclass
class BaseCalibratedJudge:
    judge_id: str
    primary_coverages: list[float]
    bad_threshold: float
    available: bool = True
    unavailable_reason: str = ""
    signal_columns_used: list[str] = field(default_factory=list)
    error_column_used: str = ""
    bad_label_column_used: str = ""
    selected_hyperparameters: dict = field(default_factory=dict)
    selected_signal_if_any: str | None = None
    fit_row_count: int = 0
    fit_scenario_count: int = 0
    fit_split_names: list[str] = field(default_factory=list)
    calibration_scenario_id_hash: str = ""
    test_scenario_id_hash: str = ""
    used_test_labels_during_fit: bool = False

    def _begin_fit(
        self,
        calibration_table: pd.DataFrame,
        signal_columns: list[str],
        error_column: str,
        bad_label_column: str,
    ) -> None:
        _ensure_no_test_rows(calibration_table)
        self.available = True
        self.unavailable_reason = ""
        self.signal_columns_used = list(signal_columns)
        self.error_column_used = error_column
        self.bad_label_column_used = bad_label_column
        self.fit_row_count = int(len(calibration_table))
        self.fit_scenario_count = int(calibration_table["scenario_id"].nunique())
        self.fit_split_names = sorted(str(value) for value in calibration_table["split"].unique())
        self.calibration_scenario_id_hash = scenario_id_hash(calibration_table["scenario_id"])
        self.used_test_labels_during_fit = False

    def _mark_unavailable(self, reason: str) -> None:
        self.available = False
        self.unavailable_reason = reason

    def set_test_scenario_hash(self, scenario_ids: pd.Series | list[str]) -> None:
        self.test_scenario_id_hash = scenario_id_hash(scenario_ids)

    def provenance(self) -> dict:
        data = {
            "judge_id": self.judge_id,
            "fit_row_count": self.fit_row_count,
            "fit_scenario_count": self.fit_scenario_count,
            "fit_split_names": self.fit_split_names,
            "signal_columns_used": self.signal_columns_used,
            "error_column_used": self.error_column_used,
            "bad_label_column_used": self.bad_label_column_used,
            "selected_hyperparameters": self.selected_hyperparameters,
            "selected_signal_if_any": self.selected_signal_if_any,
            "calibration_scenario_id_hash": self.calibration_scenario_id_hash,
            "test_scenario_id_hash": self.test_scenario_id_hash,
            "used_test_labels_during_fit": self.used_test_labels_during_fit,
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
        }
        missing = sorted(set(REQUIRED_PROVENANCE_KEYS) - set(data))
        if missing:
            raise RuntimeError(f"provenance missing keys: {missing}")
        return data

    def score(self, scenario_table: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class BestSingleSignalJudge(BaseCalibratedJudge):
    def __init__(self, primary_coverages: list[float], bad_threshold: float) -> None:
        super().__init__("best_single_signal_selected_on_calibration", primary_coverages, bad_threshold)
        self.orientation_: int = 1

    def fit(self, calibration_table: pd.DataFrame, signal_columns: list[str], error_column: str, bad_label_column: str) -> None:
        self._begin_fit(calibration_table, signal_columns, error_column, bad_label_column)
        labels = calibration_table[bad_label_column].astype(int)
        if labels.nunique() < 2:
            self._mark_unavailable("bad labels are degenerate on calibration")
            self.selected_signal_if_any = signal_columns[0] if signal_columns else None
            return
        rows = []
        for signal in signal_columns:
            raw_auc = _safe_auc(calibration_table[signal], labels)
            if np.isnan(raw_auc):
                adjusted_auc = float("nan")
                orientation = 1
            elif raw_auc < 0.5:
                adjusted_auc = 1.0 - raw_auc
                orientation = -1
            else:
                adjusted_auc = raw_auc
                orientation = 1
            scores = orientation * calibration_table[signal].to_numpy(dtype=float)
            fars = [
                _acceptance_false_accept_rate(
                    calibration_table[error_column].to_numpy(dtype=float),
                    scores,
                    self.bad_threshold,
                    coverage,
                )
                for coverage in self.primary_coverages
            ]
            rows.append(
                {
                    "signal": signal,
                    "adjusted_auc": adjusted_auc,
                    "orientation": orientation,
                    "mean_primary_far": float(np.mean(fars)),
                }
            )
        selection = pd.DataFrame(rows).sort_values(["adjusted_auc", "mean_primary_far"], ascending=[False, True], na_position="last")
        best = selection.iloc[0]
        self.selected_signal_if_any = str(best["signal"])
        self.orientation_ = int(best["orientation"])
        self.selected_hyperparameters = {
            "selection_metric": "calibration_adjusted_auroc_then_primary_far",
            "selected_signal_adjusted_auc": None if pd.isna(best["adjusted_auc"]) else float(best["adjusted_auc"]),
            "orientation": self.orientation_,
            "candidate_signals": selection.to_dict(orient="records"),
        }

    def score(self, scenario_table: pd.DataFrame) -> np.ndarray:
        if not self.available or self.selected_signal_if_any is None:
            return np.ones(len(scenario_table), dtype=float)
        return self.orientation_ * scenario_table[self.selected_signal_if_any].to_numpy(dtype=float)


class RankNormalizedLinearJudge(BaseCalibratedJudge):
    def __init__(self, primary_coverages: list[float], bad_threshold: float) -> None:
        super().__init__("rank_normalized_linear", primary_coverages, bad_threshold)
        self.orientation_: dict[str, int] = {}
        self.reference_values_: dict[str, np.ndarray] = {}
        self.weights_: dict[str, float] = {}

    def _rank_scores(self, table: pd.DataFrame) -> np.ndarray:
        scores = []
        for signal in self.signal_columns_used:
            oriented = self.orientation_.get(signal, 1) * table[signal].to_numpy(dtype=float)
            scores.append(_empirical_percentile(self.reference_values_[signal], oriented))
        return np.vstack(scores).T

    def fit(self, calibration_table: pd.DataFrame, signal_columns: list[str], error_column: str, bad_label_column: str) -> None:
        self._begin_fit(calibration_table, signal_columns, error_column, bad_label_column)
        labels = calibration_table[bad_label_column].astype(int)
        if labels.nunique() < 2:
            self._mark_unavailable("bad labels are degenerate on calibration")
            return
        for signal in signal_columns:
            auc = _safe_auc(calibration_table[signal], labels)
            self.orientation_[signal] = -1 if not np.isnan(auc) and auc < 0.5 else 1
            self.reference_values_[signal] = self.orientation_[signal] * calibration_table[signal].to_numpy(dtype=float)
        rank_matrix = self._rank_scores(calibration_table)
        grid = [0.0, 0.25, 0.5, 0.75, 1.0]
        best: tuple[float, np.ndarray] | None = None
        errors = calibration_table[error_column].to_numpy(dtype=float)
        for weights in itertools.product(grid, repeat=len(signal_columns)):
            weight_array = np.asarray(weights, dtype=float)
            if float(weight_array.sum()) <= 0.0:
                continue
            weight_array = weight_array / weight_array.sum()
            score = rank_matrix @ weight_array
            objective = float(np.mean([
                _acceptance_false_accept_rate(errors, score, self.bad_threshold, coverage)
                for coverage in self.primary_coverages
            ]))
            if best is None or objective < best[0] - 1e-12:
                best = (objective, weight_array)
        if best is None:
            self._mark_unavailable("weight grid did not contain a valid nonzero combination")
            return
        self.weights_ = {signal: float(weight) for signal, weight in zip(signal_columns, best[1])}
        self.selected_hyperparameters = {
            "weight_grid": grid,
            "weights": self.weights_,
            "orientation": self.orientation_,
            "calibration_primary_far": best[0],
        }

    def score(self, scenario_table: pd.DataFrame) -> np.ndarray:
        if not self.available or not self.weights_:
            return np.ones(len(scenario_table), dtype=float)
        rank_matrix = self._rank_scores(scenario_table)
        weights = np.asarray([self.weights_[signal] for signal in self.signal_columns_used], dtype=float)
        return rank_matrix @ weights


class CalibrationSelectedCandidateRanker(BaseCalibratedJudge):
    """Select a deployable low-coverage ranker using calibration false accepts only."""

    def __init__(self, primary_coverages: list[float], bad_threshold: float) -> None:
        super().__init__("calibration_selected_candidate_ranker", primary_coverages, bad_threshold)
        self.global_params_: dict | None = None
        self.group_params_: dict[str, dict] = {}

    @staticmethod
    def _group_key(model_id: object, scenario_type: object) -> str:
        return f"{model_id}\t{scenario_type}"

    @staticmethod
    def _has_group_columns(table: pd.DataFrame) -> bool:
        return {"model_id", "scenario_type"}.issubset(table.columns)

    def _candidate_score(self, table: pd.DataFrame, params: dict) -> np.ndarray:
        kind = params["kind"]
        if kind == "signal":
            return int(params["orientation"]) * table[str(params["signal"])].to_numpy(dtype=float)
        if kind == "combined_minmax_natural":
            columns = list(params["signal_columns"])
            scores = []
            for signal in columns:
                values = table[signal].to_numpy(dtype=float)
                low, high = params["normalizer"][signal]
                denom = max(float(high) - float(low), 1e-12)
                scores.append((values - float(low)) / denom)
            return np.mean(np.vstack(scores), axis=0)
        if kind in {"rank_mean_oriented", "rank_grid_oriented"}:
            columns = list(params["signal_columns"])
            scores = []
            for signal in columns:
                oriented = int(params["orientation"][signal]) * table[signal].to_numpy(dtype=float)
                scores.append(_empirical_percentile(np.asarray(params["reference_values"][signal], dtype=float), oriented))
            matrix = np.vstack(scores).T
            if kind == "rank_mean_oriented":
                return np.mean(matrix, axis=1)
            weights = np.asarray([params["weights"][signal] for signal in columns], dtype=float)
            return matrix @ weights
        raise ValueError(f"unknown calibrated candidate kind: {kind}")

    def _candidate_objective(self, table: pd.DataFrame, params: dict, error_column: str) -> tuple[float, float]:
        scores = self._candidate_score(table, params)
        errors = table[error_column].to_numpy(dtype=float)
        fars = [
            _acceptance_false_accept_rate(errors, scores, self.bad_threshold, coverage)
            for coverage in self.primary_coverages
        ]
        order = np.argsort(scores, kind="mergesort")
        accepted = order[: min(max(int(np.ceil(min(self.primary_coverages) * len(table))), 1), len(table))]
        return float(np.mean(fars)), float(np.mean(errors[accepted]))

    def _best_orientation(self, table: pd.DataFrame, signal: str, error_column: str) -> int:
        objectives = []
        for orientation in [1, -1]:
            params = {
                "candidate_id": f"signal:{signal}:{'high' if orientation == 1 else 'low'}",
                "kind": "signal",
                "signal": signal,
                "orientation": orientation,
            }
            objective, accepted_error = self._candidate_objective(table, params, error_column)
            objectives.append((objective, accepted_error, 0 if orientation == 1 else 1, orientation))
        return int(sorted(objectives)[0][3])

    def _candidate_priority(self, candidate_id: str) -> int:
        priorities = {
            "signal:disagreement_score:high": 0,
            "combined_minmax_natural": 1,
            "rank_grid_oriented": 2,
            "rank_mean_oriented": 3,
        }
        if candidate_id.startswith("signal:"):
            return priorities.get(candidate_id, 10)
        return priorities.get(candidate_id, 20)

    def _build_candidates(self, table: pd.DataFrame, signal_columns: list[str], error_column: str) -> list[dict]:
        candidates: list[dict] = []
        for signal in signal_columns:
            for orientation in [1, -1]:
                candidates.append(
                    {
                        "candidate_id": f"signal:{signal}:{'high' if orientation == 1 else 'low'}",
                        "kind": "signal",
                        "signal": signal,
                        "orientation": orientation,
                    }
                )

        candidates.append(
            {
                "candidate_id": "combined_minmax_natural",
                "kind": "combined_minmax_natural",
                "signal_columns": list(signal_columns),
                "normalizer": {
                    signal: [
                        float(table[signal].to_numpy(dtype=float).min()),
                        float(table[signal].to_numpy(dtype=float).max()),
                    ]
                    for signal in signal_columns
                },
            }
        )

        orientation = {signal: self._best_orientation(table, signal, error_column) for signal in signal_columns}
        reference_values = {
            signal: orientation[signal] * table[signal].to_numpy(dtype=float)
            for signal in signal_columns
        }
        rank_mean = {
            "candidate_id": "rank_mean_oriented",
            "kind": "rank_mean_oriented",
            "signal_columns": list(signal_columns),
            "orientation": dict(orientation),
            "reference_values": reference_values,
        }
        candidates.append(rank_mean)

        rank_scores = []
        for signal in signal_columns:
            oriented = reference_values[signal]
            rank_scores.append(_empirical_percentile(oriented, oriented))
        rank_matrix = np.vstack(rank_scores).T
        grid = [0.0, 0.25, 0.5, 0.75, 1.0]
        best_weights: np.ndarray | None = None
        best_objective: tuple[float, float] | None = None
        errors = table[error_column].to_numpy(dtype=float)
        for weights in itertools.product(grid, repeat=len(signal_columns)):
            weight_array = np.asarray(weights, dtype=float)
            if float(weight_array.sum()) <= 0.0:
                continue
            weight_array = weight_array / weight_array.sum()
            score = rank_matrix @ weight_array
            fars = [
                _acceptance_false_accept_rate(errors, score, self.bad_threshold, coverage)
                for coverage in self.primary_coverages
            ]
            accepted_count = min(max(int(np.ceil(min(self.primary_coverages) * len(table))), 1), len(table))
            accepted_error = float(np.mean(errors[np.argsort(score, kind="mergesort")[:accepted_count]]))
            objective = (float(np.mean(fars)), accepted_error)
            if best_objective is None or objective < best_objective:
                best_objective = objective
                best_weights = weight_array
        if best_weights is not None:
            candidates.append(
                {
                    "candidate_id": "rank_grid_oriented",
                    "kind": "rank_grid_oriented",
                    "signal_columns": list(signal_columns),
                    "orientation": dict(orientation),
                    "reference_values": reference_values,
                    "weights": {
                        signal: float(weight)
                        for signal, weight in zip(signal_columns, best_weights)
                    },
                    "weight_grid": grid,
                }
            )
        return candidates

    def _select_params(self, table: pd.DataFrame, signal_columns: list[str], error_column: str, bad_label_column: str) -> dict:
        labels = table[bad_label_column].astype(int)
        if labels.nunique() < 2:
            raise ValueError("bad labels are degenerate for candidate selection")
        rows = []
        for params in self._build_candidates(table, signal_columns, error_column):
            objective, accepted_error = self._candidate_objective(table, params, error_column)
            rows.append(
                {
                    "params": params,
                    "candidate_id": params["candidate_id"],
                    "calibration_primary_far": objective,
                    "calibration_lowest_coverage_mean_error_accepted": accepted_error,
                    "priority": self._candidate_priority(str(params["candidate_id"])),
                }
            )
        best = sorted(
            rows,
            key=lambda row: (
                row["calibration_primary_far"],
                row["priority"],
                row["calibration_lowest_coverage_mean_error_accepted"],
                row["candidate_id"],
            ),
        )[0]
        params = best["params"]
        params["calibration_primary_far"] = float(best["calibration_primary_far"])
        params["calibration_lowest_coverage_mean_error_accepted"] = float(best["calibration_lowest_coverage_mean_error_accepted"])
        return params

    @staticmethod
    def _serializable_param_summary(params: dict) -> dict:
        summary = {
            "candidate_id": params.get("candidate_id"),
            "kind": params.get("kind"),
            "calibration_primary_far": params.get("calibration_primary_far"),
            "calibration_lowest_coverage_mean_error_accepted": params.get(
                "calibration_lowest_coverage_mean_error_accepted"
            ),
        }
        if params.get("kind") == "signal":
            summary["signal"] = params.get("signal")
            summary["orientation"] = params.get("orientation")
        if params.get("kind") == "rank_grid_oriented":
            summary["weights"] = params.get("weights")
            summary["orientation"] = params.get("orientation")
        if params.get("kind") == "rank_mean_oriented":
            summary["orientation"] = params.get("orientation")
        return summary

    def fit(self, calibration_table: pd.DataFrame, signal_columns: list[str], error_column: str, bad_label_column: str) -> None:
        self._begin_fit(calibration_table, signal_columns, error_column, bad_label_column)
        try:
            self.global_params_ = self._select_params(calibration_table, signal_columns, error_column, bad_label_column)
        except ValueError as exc:
            self._mark_unavailable(str(exc))
            return

        self.group_params_ = {}
        fallback_groups = []
        if self._has_group_columns(calibration_table):
            for (model_id, scenario_type), group in calibration_table.groupby(["model_id", "scenario_type"], sort=False):
                key = self._group_key(model_id, scenario_type)
                if group[bad_label_column].astype(int).nunique() < 2:
                    fallback_groups.append(key)
                    continue
                try:
                    self.group_params_[key] = self._select_params(group, signal_columns, error_column, bad_label_column)
                except ValueError:
                    fallback_groups.append(key)

        selected_counts = Counter(
            params["candidate_id"]
            for params in [self.global_params_, *self.group_params_.values()]
            if params is not None
        )
        global_summary = self._serializable_param_summary(self.global_params_)
        group_summary = {
            key: self._serializable_param_summary(params)
            for key, params in self.group_params_.items()
        }
        self.selected_signal_if_any = (
            str(self.global_params_["signal"])
            if self.global_params_ is not None and self.global_params_.get("kind") == "signal"
            else None
        )
        self.selected_hyperparameters = {
            "selection_metric": "lowest_calibration_false_accept_rate_at_primary_coverages",
            "group_columns": ["model_id", "scenario_type"],
            "global_selection": global_summary,
            "group_selection": group_summary,
            "fallback_group_count": len(fallback_groups),
            "fallback_groups": fallback_groups,
            "selected_candidate_counts": dict(selected_counts),
            "oracle_used": False,
        }

    def score(self, scenario_table: pd.DataFrame) -> np.ndarray:
        if not self.available or self.global_params_ is None:
            return np.ones(len(scenario_table), dtype=float)
        if not self._has_group_columns(scenario_table):
            return self._candidate_score(scenario_table, self.global_params_)
        result = pd.Series(index=scenario_table.index, dtype=float)
        for (model_id, scenario_type), group in scenario_table.groupby(["model_id", "scenario_type"], sort=False):
            params = self.group_params_.get(self._group_key(model_id, scenario_type), self.global_params_)
            result.loc[group.index] = self._candidate_score(group, params)
        return result.to_numpy(dtype=float)


class LogisticCalibratedJudge(BaseCalibratedJudge):
    def __init__(self, primary_coverages: list[float], bad_threshold: float) -> None:
        super().__init__("logistic_calibrated_judge", primary_coverages, bad_threshold)
        self.model = None

    def fit(self, calibration_table: pd.DataFrame, signal_columns: list[str], error_column: str, bad_label_column: str) -> None:
        self._begin_fit(calibration_table, signal_columns, error_column, bad_label_column)
        labels = calibration_table[bad_label_column].astype(int)
        if labels.nunique() < 2:
            self._mark_unavailable("bad labels are degenerate on calibration")
            return
        self.model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
        self.model.fit(calibration_table[signal_columns], labels)
        self.selected_hyperparameters = {"class_weight": "balanced", "max_iter": 1000}

    def score(self, scenario_table: pd.DataFrame) -> np.ndarray:
        if not self.available or self.model is None:
            return np.ones(len(scenario_table), dtype=float)
        return self.model.predict_proba(scenario_table[self.signal_columns_used])[:, 1]


class IsotonicCalibratedJudge(BaseCalibratedJudge):
    def __init__(self, primary_coverages: list[float], bad_threshold: float) -> None:
        super().__init__("isotonic_calibrated_judge", primary_coverages, bad_threshold)
        self.base_judge = BestSingleSignalJudge(primary_coverages, bad_threshold)
        self.model: IsotonicRegression | None = None

    def fit(self, calibration_table: pd.DataFrame, signal_columns: list[str], error_column: str, bad_label_column: str) -> None:
        self._begin_fit(calibration_table, signal_columns, error_column, bad_label_column)
        labels = calibration_table[bad_label_column].astype(int)
        if labels.nunique() < 2:
            self._mark_unavailable("bad labels are degenerate on calibration")
            return
        self.base_judge.fit(calibration_table, signal_columns, error_column, bad_label_column)
        if not self.base_judge.available:
            self._mark_unavailable(self.base_judge.unavailable_reason)
            return
        base_score = self.base_judge.score(calibration_table)
        if len(np.unique(base_score)) < 3:
            self._mark_unavailable("isotonic base score has fewer than three unique values")
            return
        self.model = IsotonicRegression(out_of_bounds="clip")
        self.model.fit(base_score, labels)
        self.selected_signal_if_any = self.base_judge.selected_signal_if_any
        self.selected_hyperparameters = {
            "base_score": "best_single_signal_selected_on_calibration",
            "base_provenance": self.base_judge.provenance(),
        }

    def score(self, scenario_table: pd.DataFrame) -> np.ndarray:
        if not self.available or self.model is None:
            return np.ones(len(scenario_table), dtype=float)
        return np.asarray(self.model.predict(self.base_judge.score(scenario_table)), dtype=float)


class QuantileRuleJudge(BaseCalibratedJudge):
    def __init__(self, primary_coverages: list[float], bad_threshold: float) -> None:
        super().__init__("quantile_rule_judge", primary_coverages, bad_threshold)
        self.base_judge = BestSingleSignalJudge(primary_coverages, bad_threshold)
        self.quantile_thresholds_: dict[str, float] = {}

    def fit(self, calibration_table: pd.DataFrame, signal_columns: list[str], error_column: str, bad_label_column: str) -> None:
        self._begin_fit(calibration_table, signal_columns, error_column, bad_label_column)
        self.base_judge.fit(calibration_table, signal_columns, error_column, bad_label_column)
        if not self.base_judge.available:
            self._mark_unavailable(self.base_judge.unavailable_reason)
            return
        score = self.base_judge.score(calibration_table)
        self.quantile_thresholds_ = {
            str(coverage): float(np.quantile(score, coverage))
            for coverage in self.primary_coverages
        }
        self.selected_signal_if_any = self.base_judge.selected_signal_if_any
        self.selected_hyperparameters = {
            "base_score": "best_single_signal_selected_on_calibration",
            "quantile_thresholds": self.quantile_thresholds_,
            "base_provenance": self.base_judge.provenance(),
        }

    def score(self, scenario_table: pd.DataFrame) -> np.ndarray:
        if not self.available:
            return np.ones(len(scenario_table), dtype=float)
        return self.base_judge.score(scenario_table)


class ConservativeLowCoverageJudge(BaseCalibratedJudge):
    def __init__(self, primary_coverages: list[float], bad_threshold: float) -> None:
        super().__init__("conservative_low_coverage_judge", primary_coverages, bad_threshold)
        self.thresholds_: dict[str, float] = {}
        self.quantile_: float = 0.2

    def _risk(self, table: pd.DataFrame) -> np.ndarray:
        risks = []
        for signal in self.signal_columns_used:
            threshold = self.thresholds_[signal]
            values = table[signal].to_numpy(dtype=float)
            scale = max(abs(threshold), float(np.std(values)), 1e-6)
            risks.append(np.maximum((values - threshold) / scale, 0.0))
        return np.mean(np.vstack(risks), axis=0)

    def fit(self, calibration_table: pd.DataFrame, signal_columns: list[str], error_column: str, bad_label_column: str) -> None:
        self._begin_fit(calibration_table, signal_columns, error_column, bad_label_column)
        best: tuple[float, float, dict[str, float]] | None = None
        errors = calibration_table[error_column].to_numpy(dtype=float)
        for quantile in [0.10, 0.20]:
            thresholds = {
                signal: float(np.quantile(calibration_table[signal].to_numpy(dtype=float), quantile))
                for signal in signal_columns
            }
            self.thresholds_ = thresholds
            risk = self._risk(calibration_table)
            accepted_fraction = float(np.mean(risk <= 1e-12))
            if accepted_fraction <= 0.0:
                objective = 1.0
            else:
                objective = float(np.mean(calibration_table.loc[risk <= 1e-12, error_column].to_numpy(dtype=float) > self.bad_threshold))
            objective += abs(accepted_fraction - min(self.primary_coverages)) * 0.05
            if best is None or objective < best[0] - 1e-12:
                best = (objective, quantile, thresholds)
        if best is None:
            self._mark_unavailable("no conservative thresholds could be selected")
            return
        self.quantile_ = best[1]
        self.thresholds_ = best[2]
        self.selected_hyperparameters = {
            "quantile": self.quantile_,
            "thresholds": self.thresholds_,
            "calibration_objective": best[0],
        }

    def score(self, scenario_table: pd.DataFrame) -> np.ndarray:
        if not self.available or not self.thresholds_:
            return np.ones(len(scenario_table), dtype=float)
        return self._risk(scenario_table)


CALIBRATED_JUDGE_IDS = [
    "best_single_signal_selected_on_calibration",
    "rank_normalized_linear",
    "calibration_selected_candidate_ranker",
    "logistic_calibrated_judge",
    "isotonic_calibrated_judge",
    "quantile_rule_judge",
    "conservative_low_coverage_judge",
]


def make_calibrated_judges(primary_coverages: list[float], bad_threshold: float) -> list[BaseCalibratedJudge]:
    return [
        BestSingleSignalJudge(primary_coverages, bad_threshold),
        RankNormalizedLinearJudge(primary_coverages, bad_threshold),
        CalibrationSelectedCandidateRanker(primary_coverages, bad_threshold),
        LogisticCalibratedJudge(primary_coverages, bad_threshold),
        IsotonicCalibratedJudge(primary_coverages, bad_threshold),
        QuantileRuleJudge(primary_coverages, bad_threshold),
        ConservativeLowCoverageJudge(primary_coverages, bad_threshold),
    ]


def provenance_json_dumps(provenance: list[dict]) -> str:
    return json.dumps(provenance, indent=2, sort_keys=True)
