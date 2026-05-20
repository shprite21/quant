"""Tests for feature engineering."""

from __future__ import annotations

from src.features import FeatureEngineer


def test_feature_engineer_creates_expected_columns(sample_ohlcv) -> None:
    engineer = FeatureEngineer(lag_features_by=1)
    features = engineer.build_features(sample_ohlcv, dropna=False)

    expected_columns = {
        "daily_return",
        "momentum_5",
        "rolling_volatility_20",
        "rsi_14",
        "macd",
        "atr_percent",
        "bollinger_bandwidth",
        "adx",
        "relative_volume_20",
        "trend_strength",
    }
    assert expected_columns.issubset(set(features.columns))
    assert features["daily_return"].isna().iloc[0]
    assert features["momentum_20"].notna().sum() > 0
