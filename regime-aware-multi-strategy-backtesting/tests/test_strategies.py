from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.momentum import MomentumStrategy
from src.strategies.pairs_trading import PairsTradingStrategy


def test_momentum_strategy_generates_directional_signals() -> None:
    index = pd.date_range("2023-01-01", periods=6, freq="D")
    prices = pd.DataFrame(
        {
            "UP": [1, 2, 3, 4, 5, 6],
            "DOWN": [6, 5, 4, 3, 2, 1],
        },
        index=index,
    )
    strategy = MomentumStrategy(short_window=2, long_window=3, assets=["UP", "DOWN"])
    output = strategy.generate_positions(prices)

    assert output.positions.loc[index[-1], "UP"] == 1.0
    assert output.positions.loc[index[-1], "DOWN"] == -1.0


def test_mean_reversion_strategy_enters_and_can_flip_direction() -> None:
    index = pd.date_range("2023-01-01", periods=8, freq="D")
    prices = pd.DataFrame(
        {
            "X": [100, 100, 100, 95, 94, 99, 100, 100],
        },
        index=index,
    )
    strategy = MeanReversionStrategy(window=3, entry_threshold=1.0, exit_threshold=0.2, assets=["X"])
    output = strategy.generate_positions(prices)

    assert output.positions["X"].max() == 1.0
    assert output.positions.loc[index[-1], "X"] <= 0.0


def test_pairs_trading_strategy_builds_two_asset_exposures() -> None:
    index = pd.date_range("2022-01-01", periods=120, freq="D")
    asset_x = pd.Series(np.linspace(100, 120, len(index)), index=index)
    asset_y = asset_x * 1.02 + np.sin(np.linspace(0, 10, len(index)))
    asset_y.iloc[80:85] = asset_y.iloc[80:85] + 4.0

    prices = pd.DataFrame({"AAPL": asset_x, "MSFT": asset_y}, index=index)
    strategy = PairsTradingStrategy(
        asset_x="AAPL",
        asset_y="MSFT",
        lookback_window=30,
        zscore_window=10,
        entry_threshold=1.0,
        exit_threshold=0.2,
    )
    output = strategy.generate_positions(prices)

    assert {"AAPL", "MSFT"}.issubset(output.positions.columns)
    assert output.positions.abs().sum().sum() > 0.0
