"""Unsupervised market regime detection models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


@dataclass(slots=True)
class RegimeDetectionResult:
    """Container for regime assignments and state diagnostics."""

    assignments: pd.DataFrame
    transition_matrix: pd.DataFrame
    duration_stats: pd.DataFrame
    regime_summary: pd.DataFrame


class RegimeDetector:
    """Detect market regimes using HMM or GMM models."""

    def __init__(
        self,
        method: str = "hmm",
        n_regimes: int = 3,
        covariance_type: str = "full",
        n_iter: int = 200,
        random_state: int = 42,
    ) -> None:
        self.method = method.lower()
        self.n_regimes = n_regimes
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state

        self.model: GaussianHMM | GaussianMixture | None = None
        self.scaler = StandardScaler()
        self.feature_columns_: list[str] = []
        self.state_order_: list[int] = []
        self.state_mapping_: dict[int, int] = {}
        self.transition_matrix_: pd.DataFrame | None = None
        self.duration_stats_: pd.DataFrame | None = None
        self.regime_summary_: pd.DataFrame | None = None

    def fit(self, data: pd.DataFrame, feature_columns: list[str]) -> "RegimeDetector":
        """Fit the chosen unsupervised model on the supplied features."""

        clean_data, matrix = self._prepare_features(data, feature_columns)
        self.feature_columns_ = feature_columns.copy()
        self.model = self._build_model()
        self.model.fit(matrix)

        raw_labels, raw_probabilities = self._infer_states(matrix)
        self.state_order_ = self._determine_state_order(clean_data, raw_labels)
        self.state_mapping_ = {
            original_state: ordered_state
            for ordered_state, original_state in enumerate(self.state_order_)
        }

        ordered_labels = self._map_labels(raw_labels)
        ordered_probabilities = self._reorder_probabilities(raw_probabilities)
        self.transition_matrix_ = self._build_transition_matrix(ordered_labels)
        self.duration_stats_ = self._calculate_duration_stats(ordered_labels)
        self.regime_summary_ = self._summarize_regimes(clean_data, ordered_labels, ordered_probabilities)
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict regime labels and state probabilities for new observations."""

        if self.model is None or not self.feature_columns_:
            raise ValueError("The regime detector must be fitted before prediction.")

        clean_data, matrix = self._prepare_features(data, self.feature_columns_)
        raw_labels, raw_probabilities = self._infer_states(matrix)
        labels = self._map_labels(raw_labels)
        probabilities = self._reorder_probabilities(raw_probabilities)
        assignments = self._format_assignments(clean_data.index, labels, probabilities)

        full_assignments = pd.DataFrame(index=data.index)
        full_assignments = full_assignments.join(assignments, how="left")
        full_assignments["regime_duration"] = self._calculate_current_run_lengths(
            full_assignments["regime_id"]
        )
        return full_assignments

    def fit_predict(
        self,
        data: pd.DataFrame,
        feature_columns: list[str],
    ) -> RegimeDetectionResult:
        """Fit the model and return assignments plus diagnostics."""

        self.fit(data, feature_columns)
        assignments = self.predict(data)
        return RegimeDetectionResult(
            assignments=assignments,
            transition_matrix=self.transition_matrix_.copy()
            if self.transition_matrix_ is not None
            else pd.DataFrame(),
            duration_stats=self.duration_stats_.copy()
            if self.duration_stats_ is not None
            else pd.DataFrame(),
            regime_summary=self.regime_summary_.copy()
            if self.regime_summary_ is not None
            else pd.DataFrame(),
        )

    def _build_model(self) -> GaussianHMM | GaussianMixture:
        if self.method == "hmm":
            return GaussianHMM(
                n_components=self.n_regimes,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=self.random_state,
            )
        if self.method == "gmm":
            return GaussianMixture(
                n_components=self.n_regimes,
                covariance_type=self.covariance_type,
                max_iter=self.n_iter,
                random_state=self.random_state,
            )
        raise ValueError("method must be either 'hmm' or 'gmm'.")

    def _prepare_features(
        self,
        data: pd.DataFrame,
        feature_columns: list[str],
    ) -> tuple[pd.DataFrame, np.ndarray]:
        subset = data[feature_columns].copy()
        subset = subset.replace([np.inf, -np.inf], np.nan).dropna()
        if subset.empty:
            raise ValueError("No rows remain after dropping missing regime features.")
        matrix = subset.to_numpy(dtype=float)
        matrix = self.scaler.fit_transform(matrix) if self.model is None else self.scaler.transform(matrix)
        return data.loc[subset.index].copy(), matrix

    def _infer_states(self, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise ValueError("Model has not been initialized.")
        if self.method == "hmm":
            labels = self.model.predict(matrix)
            probabilities = self.model.predict_proba(matrix)
        else:
            labels = self.model.predict(matrix)
            probabilities = self.model.predict_proba(matrix)
        return labels, probabilities

    def _determine_state_order(self, data: pd.DataFrame, labels: np.ndarray) -> list[int]:
        ordering_feature = "daily_return" if "daily_return" in data.columns else self.feature_columns_[0]
        ordering = (
            pd.DataFrame({"state": labels, ordering_feature: data[ordering_feature].to_numpy()})
            .groupby("state")[ordering_feature]
            .mean()
            .sort_values()
        )
        return ordering.index.to_list()

    def _map_labels(self, labels: np.ndarray) -> np.ndarray:
        return np.array([self.state_mapping_[label] for label in labels], dtype=int)

    def _reorder_probabilities(self, probabilities: np.ndarray) -> np.ndarray:
        return probabilities[:, self.state_order_.copy()]

    def _build_transition_matrix(self, labels: np.ndarray) -> pd.DataFrame:
        if self.method == "hmm" and self.model is not None and hasattr(self.model, "transmat_"):
            matrix = np.asarray(self.model.transmat_)
            matrix = matrix[np.ix_(self.state_order_, self.state_order_)]
        else:
            matrix = np.zeros((self.n_regimes, self.n_regimes), dtype=float)
            for current_state, next_state in zip(labels[:-1], labels[1:]):
                matrix[current_state, next_state] += 1
            row_totals = matrix.sum(axis=1, keepdims=True)
            matrix = np.divide(
                matrix,
                row_totals,
                out=np.zeros_like(matrix),
                where=row_totals != 0,
            )
        labels_index = [f"regime_{state}" for state in range(self.n_regimes)]
        return pd.DataFrame(matrix, index=labels_index, columns=labels_index)

    def _calculate_duration_stats(self, labels: np.ndarray) -> pd.DataFrame:
        durations: dict[int, list[int]] = {state: [] for state in range(self.n_regimes)}
        if len(labels) == 0:
            return pd.DataFrame()
        current_state = int(labels[0])
        current_length = 1
        for label in labels[1:]:
            if int(label) == current_state:
                current_length += 1
            else:
                durations[current_state].append(current_length)
                current_state = int(label)
                current_length = 1
        durations[current_state].append(current_length)

        summary_rows = []
        for state, values in durations.items():
            state_values = pd.Series(values, dtype=float)
            summary_rows.append(
                {
                    "regime_id": state,
                    "occurrences": int(state_values.count()),
                    "mean_duration": float(state_values.mean()) if not state_values.empty else 0.0,
                    "median_duration": float(state_values.median()) if not state_values.empty else 0.0,
                    "max_duration": float(state_values.max()) if not state_values.empty else 0.0,
                }
            )
        return pd.DataFrame(summary_rows)

    def _summarize_regimes(
        self,
        data: pd.DataFrame,
        labels: np.ndarray,
        probabilities: np.ndarray,
    ) -> pd.DataFrame:
        summary = data.copy()
        summary["regime_id"] = labels
        summary["regime_probability"] = probabilities.max(axis=1)

        numeric_columns = summary.select_dtypes(include=[np.number]).columns.tolist()
        numeric_columns = [column for column in numeric_columns if column != "regime_id"]
        grouped = summary.groupby("regime_id")[numeric_columns].mean()
        grouped["count"] = summary.groupby("regime_id").size()
        return grouped.reset_index()

    def _format_assignments(
        self,
        index: pd.Index,
        labels: np.ndarray,
        probabilities: np.ndarray,
    ) -> pd.DataFrame:
        assignments = pd.DataFrame(index=index)
        assignments["regime_id"] = labels
        assignments["regime_probability"] = probabilities.max(axis=1)
        for state in range(self.n_regimes):
            assignments[f"regime_probability_{state}"] = probabilities[:, state]
        assignments["state_probability"] = assignments["regime_probability"]
        return assignments

    @staticmethod
    def _calculate_current_run_lengths(labels: pd.Series) -> pd.Series:
        current_length = 0
        previous_state: float | int | None = None
        run_lengths: list[float] = []
        for value in labels.tolist():
            if pd.isna(value):
                current_length = 0
                previous_state = None
                run_lengths.append(np.nan)
                continue
            if previous_state is None or value != previous_state:
                current_length = 1
            else:
                current_length += 1
            previous_state = value
            run_lengths.append(float(current_length))
        return pd.Series(run_lengths, index=labels.index, name="regime_duration")
