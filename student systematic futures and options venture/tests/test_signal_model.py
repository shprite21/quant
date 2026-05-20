"""Tests for supervised directional models."""

from __future__ import annotations

from src.features import FeatureEngineer
from src.signal_model import DirectionalSignalModel, create_targets


def test_signal_model_trains_random_forest_classifier(sample_ohlcv) -> None:
    features = FeatureEngineer(lag_features_by=1).build_features(sample_ohlcv, dropna=True)
    dataset = create_targets(features, horizon=5, return_threshold=0.01).dropna()

    feature_columns = [
        "momentum_5",
        "rolling_volatility_20",
        "rsi_14",
        "macd_hist",
        "atr_percent",
    ]
    model = DirectionalSignalModel(model_name="random_forest", task_type="binary")
    model.fit(dataset.iloc[:140], feature_columns=feature_columns, target_column="target_binary")
    predictions = model.make_prediction_frame(dataset.iloc[140:])
    importance = model.get_feature_importance()

    assert "up_probability" in predictions.columns
    assert len(importance) == len(feature_columns)
