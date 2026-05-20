"""Shared test fixtures for the trading research repository."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def sample_ohlcv() -> pd.DataFrame:
    """Create a deterministic OHLCV sample for tests."""

    rng = np.random.default_rng(42)
    index = pd.date_range("2020-01-01", periods=220, freq="B")
    close = 100 + np.cumsum(rng.normal(loc=0.15, scale=1.0, size=len(index)))
    open_price = np.roll(close, 1)
    open_price[0] = close[0]
    open_price = open_price * (1 + rng.normal(loc=0.0, scale=0.002, size=len(index)))
    high = np.maximum(open_price, close) * (1 + rng.uniform(0.001, 0.015, size=len(index)))
    low = np.minimum(open_price, close) * (1 - rng.uniform(0.001, 0.015, size=len(index)))
    volume = rng.integers(100_000, 500_000, size=len(index))

    return pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )
