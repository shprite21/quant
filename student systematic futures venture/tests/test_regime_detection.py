"""Tests for unsupervised regime detection."""

from __future__ import annotations

from src.features import FeatureEngineer
from src.regime_detection import RegimeDetector


def test_regime_detector_fit_predict_returns_diagnostics(sample_ohlcv) -> None:
    features = FeatureEngineer(lag_features_by=1).build_features(sample_ohlcv, dropna=True)
    detector = RegimeDetector(method="gmm", n_regimes=2, random_state=42)
    result = detector.fit_predict(
        features,
        feature_columns=["daily_return", "rolling_volatility_20", "atr_percent"],
    )

    assert {"regime_id", "regime_probability", "regime_duration"}.issubset(
        result.assignments.columns
    )
    assert result.transition_matrix.shape == (2, 2)
    assert not result.duration_stats.empty
