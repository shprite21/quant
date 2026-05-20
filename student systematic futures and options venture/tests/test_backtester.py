"""Tests for walk-forward backtesting."""

from __future__ import annotations

from src.backtester import WalkForwardBacktester
from src.features import FeatureEngineer
from src.signal_model import create_targets


def test_walk_forward_backtester_runs_end_to_end(sample_ohlcv) -> None:
    features = FeatureEngineer(lag_features_by=1).build_features(sample_ohlcv, dropna=True)
    dataset = create_targets(features, horizon=3, return_threshold=0.005).dropna()

    feature_columns = [
        "momentum_5",
        "rolling_volatility_10",
        "rolling_volatility_20",
        "rsi_14",
        "macd_hist",
        "atr_percent",
        "trend_strength",
    ]
    backtester = WalkForwardBacktester(
        model_name="random_forest",
        task_type="binary",
        train_window=80,
        test_window=20,
        step_size=20,
        transaction_cost_bps=1.0,
        slippage_bps=0.5,
    )

    result = backtester.run(
        dataset,
        feature_columns=feature_columns,
        target_column="target_binary",
        price_column="close",
        returns_column="daily_return",
    )

    assert "strategy_return" in result.predictions.columns
    assert "strategy" in result.performance_summary.index
    assert not result.fold_metrics.empty
