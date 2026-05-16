"""Tests for market data loading utilities."""

from __future__ import annotations

import pandas as pd

from src.data_loader import MarketDataLoader


def test_clean_ohlcv_standardizes_columns_and_index() -> None:
    loader = MarketDataLoader()
    raw = pd.DataFrame(
        {
            "Date": ["2024-01-02", "2024-01-01"],
            "Open": [101, 100],
            "High": [102, 101],
            "Low": [99, 98],
            "Close": [100.5, 100.0],
            "Volume": [1000, 1200],
        }
    )

    cleaned = loader.clean_ohlcv(raw, date_col="Date", symbol="TEST")

    assert cleaned.index.is_monotonic_increasing
    assert list(cleaned.columns[:6]) == [
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
    ]
    assert cleaned["symbol"].eq("TEST").all()
