from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MarketDataBundle:
    """Container for a named market dataset."""

    symbol: str
    frame: pd.DataFrame


def standardize_ohlcv_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize common market-data column names to a predictable schema."""

    renamed = frame.rename(columns=lambda value: str(value).strip().lower().replace(" ", "_"))
    aliases = {
        "adj_close": "adj_close",
        "adjclose": "adj_close",
        "timestamp": "date",
        "datetime": "date",
    }
    renamed = renamed.rename(columns=aliases)
    return renamed


def load_ohlcv_csv(path: str | Path, date_col: str = "date") -> pd.DataFrame:
    """Load an OHLCV CSV into a date-indexed pandas DataFrame."""

    frame = pd.read_csv(path)
    frame = standardize_ohlcv_columns(frame)

    if date_col not in frame.columns:
        if "date" in frame.columns:
            date_col = "date"
        else:
            raise ValueError(f"Could not find a date column in {path}.")

    frame[date_col] = pd.to_datetime(frame[date_col])
    frame = frame.sort_values(date_col).set_index(date_col)
    return frame


def download_yfinance(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance using yfinance."""

    import yfinance as yf

    frame = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=False, progress=False)

    if frame.empty:
        raise ValueError(f"No data returned for symbol '{symbol}'.")

    frame = standardize_ohlcv_columns(frame.reset_index())
    return frame.set_index("date").sort_index()


def load_many_csv(paths: Mapping[str, str | Path]) -> dict[str, pd.DataFrame]:
    """Load multiple CSV files keyed by ticker or asset name."""

    return {symbol: load_ohlcv_csv(path) for symbol, path in paths.items()}


def align_close_prices(
    frames: Mapping[str, pd.DataFrame],
    price_col: str = "close",
    join: str = "inner",
) -> pd.DataFrame:
    """Align close-price series from several assets into a single panel."""

    close_prices = []

    for symbol, frame in frames.items():
        if price_col not in frame.columns:
            raise ValueError(f"'{price_col}' was not found for symbol '{symbol}'.")
        close_prices.append(frame[[price_col]].rename(columns={price_col: symbol}))

    return pd.concat(close_prices, axis=1, join=join).sort_index()


def compute_log_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Compute log returns from a price series or price panel."""

    return np.log(prices).diff()


def resample_ohlcv(frame: pd.DataFrame, rule: str = "W-FRI") -> pd.DataFrame:
    """Resample daily OHLCV data to a lower frequency."""

    aggregation_map = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "adj_close": "last",
        "volume": "sum",
    }
    available_columns = {column: func for column, func in aggregation_map.items() if column in frame.columns}
    return frame.resample(rule).agg(available_columns).dropna(how="all")


def chronological_split(
    frame: pd.DataFrame,
    train_size: float = 0.6,
    validation_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a time series into train, validation, and test segments."""

    if not 0 < train_size < 1:
        raise ValueError("train_size must be between 0 and 1.")
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1.")
    if train_size + validation_size >= 1:
        raise ValueError("train_size + validation_size must be less than 1.")

    total_rows = len(frame)
    train_end = int(total_rows * train_size)
    validation_end = int(total_rows * (train_size + validation_size))

    train = frame.iloc[:train_end]
    validation = frame.iloc[train_end:validation_end]
    test = frame.iloc[validation_end:]
    return train, validation, test

