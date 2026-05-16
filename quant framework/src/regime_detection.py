from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:
    GaussianHMM = None


@dataclass(frozen=True)
class RegimeSummary:
    """Human-readable statistics for a fitted market state."""

    regime_id: int
    regime_label: str
    avg_return: float
    avg_volatility: float
    avg_trend: float
    avg_confidence: float
    sample_count: int


def _prepare_feature_frame(feature_frame: pd.DataFrame) -> pd.DataFrame:
    clean = feature_frame.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        raise ValueError("The feature matrix is empty after dropping missing values.")
    return clean


def _summarize_states(
    clean: pd.DataFrame,
    regime_ids: np.ndarray,
    probabilities: np.ndarray,
    return_col: str,
    vol_col: str,
    trend_col: str,
) -> pd.DataFrame:
    required = {return_col, vol_col, trend_col}
    missing = required.difference(clean.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Missing required regime summary columns: {missing_text}.")

    assignments = clean[[return_col, vol_col, trend_col]].copy()
    assignments["regime_id"] = regime_ids
    assignments["probability"] = probabilities

    return assignments.groupby("regime_id").agg(
        avg_return=(return_col, "mean"),
        avg_volatility=(vol_col, "mean"),
        avg_trend=(trend_col, "mean"),
        avg_confidence=("probability", "mean"),
        sample_count=("probability", "size"),
    )


def _build_regime_map(stats: pd.DataFrame) -> dict[int, str]:
    return_band = max(stats["avg_return"].abs().median() * 0.25, 1e-5)
    vol_cutoff = stats["avg_volatility"].median()
    trend_cutoff = stats["avg_trend"].median()

    regime_map: dict[int, str] = {}
    label_counts: dict[str, int] = {}

    for regime_id, row in stats.iterrows():
        if row["avg_return"] > return_band:
            direction = "bullish"
        elif row["avg_return"] < -return_band:
            direction = "bearish"
        else:
            direction = "neutral"

        volatility = "high_volatility" if row["avg_volatility"] >= vol_cutoff else "low_volatility"
        style = "trending" if row["avg_trend"] >= trend_cutoff else "mean_reverting"
        base_label = f"{direction}_{volatility}_{style}"

        label_counts[base_label] = label_counts.get(base_label, 0) + 1
        if label_counts[base_label] > 1:
            base_label = f"{base_label}_state_{regime_id}"

        regime_map[int(regime_id)] = base_label

    return regime_map


class _BaseRegimeDetector:
    """Shared logic for unsupervised regime detectors."""

    model_name: str = "unspecified"

    def __init__(self, n_regimes: int = 4, random_state: int = 42) -> None:
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names_: list[str] = []
        self.regime_map_: dict[int, str] = {}
        self.regime_stats_: pd.DataFrame | None = None
        self.state_columns_: list[str] = []

    def fit(
        self,
        feature_frame: pd.DataFrame,
        return_col: str = "log_return_1d",
        vol_col: str = "realized_vol_20d",
        trend_col: str = "sma_gap_20d",
    ) -> "_BaseRegimeDetector":
        """Fit the regime model on a clean feature matrix."""

        clean = _prepare_feature_frame(feature_frame)
        self.feature_names_ = list(clean.columns)

        scaled = self.scaler.fit_transform(clean)
        self._fit_model(scaled)
        regime_ids, probability_frame = self._infer_states(scaled, clean.index)

        stats = _summarize_states(
            clean=clean,
            regime_ids=regime_ids,
            probabilities=probability_frame.max(axis=1).to_numpy(),
            return_col=return_col,
            vol_col=vol_col,
            trend_col=trend_col,
        )
        self.regime_map_ = _build_regime_map(stats)
        self.regime_stats_ = stats.assign(
            regime_label=lambda frame: frame.index.map(self.regime_map_),
        )
        self.state_columns_ = list(probability_frame.columns)
        return self

    def predict(self, feature_frame: pd.DataFrame) -> pd.DataFrame:
        """Predict regime assignments for new observations."""

        self._check_is_fitted()
        clean = _prepare_feature_frame(feature_frame[self.feature_names_])
        scaled = self.scaler.transform(clean)
        regime_ids, probability_frame = self._infer_states(scaled, clean.index)
        assignments = self._build_assignments(clean.index, regime_ids, probability_frame)
        return assignments.reindex(feature_frame.index)

    def fit_predict(
        self,
        feature_frame: pd.DataFrame,
        return_col: str = "log_return_1d",
        vol_col: str = "realized_vol_20d",
        trend_col: str = "sma_gap_20d",
    ) -> pd.DataFrame:
        """Fit the model and return in-sample regime assignments."""

        self.fit(feature_frame, return_col=return_col, vol_col=vol_col, trend_col=trend_col)
        return self.predict(feature_frame)

    def summarize_regimes(self) -> list[RegimeSummary]:
        """Return regime summaries as dataclass instances."""

        self._check_is_fitted()
        assert self.regime_stats_ is not None

        summaries = []
        for regime_id, row in self.regime_stats_.iterrows():
            summaries.append(
                RegimeSummary(
                    regime_id=int(regime_id),
                    regime_label=str(row["regime_label"]),
                    avg_return=float(row["avg_return"]),
                    avg_volatility=float(row["avg_volatility"]),
                    avg_trend=float(row["avg_trend"]),
                    avg_confidence=float(row["avg_confidence"]),
                    sample_count=int(row["sample_count"]),
                )
            )
        return summaries

    def _build_assignments(
        self,
        index: pd.Index,
        regime_ids: np.ndarray,
        probability_frame: pd.DataFrame,
    ) -> pd.DataFrame:
        assignments = pd.DataFrame(
            {
                "regime_id": regime_ids,
                "regime_probability": probability_frame.max(axis=1),
                "regime_label": pd.Series(regime_ids, index=index).map(self.regime_map_),
            },
            index=index,
        )
        return assignments.join(probability_frame)

    def _fit_model(self, scaled: np.ndarray) -> None:
        raise NotImplementedError

    def _infer_states(self, scaled: np.ndarray, index: pd.Index) -> tuple[np.ndarray, pd.DataFrame]:
        raise NotImplementedError

    def _check_is_fitted(self) -> None:
        if not self.feature_names_:
            raise ValueError("Fit the RegimeDetector before calling predict or summarize_regimes.")


class HiddenMarkovRegimeDetector(_BaseRegimeDetector):
    """Primary regime detector based on a Gaussian Hidden Markov Model."""

    model_name = "hmm"

    def __init__(
        self,
        n_regimes: int = 4,
        covariance_type: str = "diag",
        n_iter: int = 200,
        random_state: int = 42,
    ) -> None:
        super().__init__(n_regimes=n_regimes, random_state=random_state)
        if GaussianHMM is None:
            raise ImportError(
                "hmmlearn is required for HiddenMarkovRegimeDetector. Install it with `pip install hmmlearn`."
            )
        self.model = GaussianHMM(
            n_components=n_regimes,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state,
        )

    def transition_matrix(self) -> pd.DataFrame:
        """Return the fitted HMM transition matrix."""

        self._check_is_fitted()
        labels = [f"state_{state}_{self.regime_map_.get(state, 'unmapped')}" for state in range(self.n_regimes)]
        return pd.DataFrame(self.model.transmat_, index=labels, columns=labels)

    def _fit_model(self, scaled: np.ndarray) -> None:
        self.model.fit(scaled)

    def _infer_states(self, scaled: np.ndarray, index: pd.Index) -> tuple[np.ndarray, pd.DataFrame]:
        regime_ids = self.model.predict(scaled)
        _, posterior = self.model.score_samples(scaled)
        probability_frame = pd.DataFrame(
            posterior,
            index=index,
            columns=[f"state_probability_{state}" for state in range(self.n_regimes)],
        )
        return regime_ids, probability_frame


class GaussianMixtureRegimeDetector(_BaseRegimeDetector):
    """Flexible baseline regime detector using a Gaussian mixture model."""

    model_name = "gmm"

    def __init__(
        self,
        n_regimes: int = 4,
        covariance_type: str = "full",
        random_state: int = 42,
    ) -> None:
        super().__init__(n_regimes=n_regimes, random_state=random_state)
        self.model = GaussianMixture(
            n_components=n_regimes,
            covariance_type=covariance_type,
            random_state=random_state,
        )

    def _fit_model(self, scaled: np.ndarray) -> None:
        self.model.fit(scaled)

    def _infer_states(self, scaled: np.ndarray, index: pd.Index) -> tuple[np.ndarray, pd.DataFrame]:
        regime_ids = self.model.predict(scaled)
        posterior = self.model.predict_proba(scaled)
        probability_frame = pd.DataFrame(
            posterior,
            index=index,
            columns=[f"state_probability_{state}" for state in range(self.n_regimes)],
        )
        return regime_ids, probability_frame


RegimeDetector = HiddenMarkovRegimeDetector
