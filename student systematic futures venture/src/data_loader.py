"""Utilities for loading and standardizing market data."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf

PathLike = str | Path

_COLUMN_ALIASES = {
    "date": "date",
    "datetime": "date",
    "timestamp": "date",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "adjclose": "adj_close",
    "adj_close": "adj_close",
    "adjustedclose": "adj_close",
    "volume": "volume",
    "openinterest": "open_interest",
    "oi": "open_interest",
}

_REQUIRED_PRICE_COLUMNS = ("open", "high", "low", "close")


def _normalize_column_name(column_name: str) -> str:
    """Normalize a column name for alias lookup."""

    return "".join(character for character in column_name.lower() if character.isalnum())


class MarketDataLoader:
    """Load OHLCV data from local CSV files and Yahoo Finance."""

    def __init__(self, default_interval: str = "1d", auto_adjust: bool = False) -> None:
        self.default_interval = default_interval
        self.auto_adjust = auto_adjust

    def load_csv(
        self,
        path: PathLike,
        date_col: str | None = None,
        symbol: str | None = None,
        fill_method: str = "ffill",
    ) -> pd.DataFrame:
        """Load and clean OHLCV data from a CSV file."""

        csv_path = Path(path)
        frame = pd.read_csv(csv_path)
        return self.clean_ohlcv(
            frame,
            date_col=date_col,
            symbol=symbol,
            fill_method=fill_method,
        )

    def download_yfinance(
        self,
        ticker: str,
        start: str | None = None,
        end: str | None = None,
        interval: str | None = None,
        fill_method: str = "ffill",
    ) -> pd.DataFrame:
        """Download OHLCV data for a ticker from Yahoo Finance."""

        raw = yf.download(
            tickers=ticker,
            start=start,
            end=end,
            interval=interval or self.default_interval,
            auto_adjust=self.auto_adjust,
            progress=False,
            group_by="column",
        )
        if raw.empty:
            raise ValueError(f"No data returned by yfinance for ticker '{ticker}'.")
        if isinstance(raw.columns, pd.MultiIndex):
            if len(raw.columns.get_level_values(-1).unique()) == 1:
                raw.columns = raw.columns.get_level_values(0)
            else:
                raw.columns = [
                    "_".join(str(part) for part in column if str(part))
                    for column in raw.columns.to_flat_index()
                ]
        raw = raw.reset_index()
        return self.clean_ohlcv(
            raw,
            date_col="Date",
            symbol=ticker,
            fill_method=fill_method,
        )

    def load_many_csv(
        self,
        paths: dict[str, PathLike],
        date_col: str | None = None,
        fill_method: str = "ffill",
    ) -> dict[str, pd.DataFrame]:
        """Load multiple symbol CSV files into a dictionary of data frames."""

        return {
            symbol: self.load_csv(
                path=path,
                date_col=date_col,
                symbol=symbol,
                fill_method=fill_method,
            )
            for symbol, path in paths.items()
        }

    def clean_ohlcv(
        self,
        data: pd.DataFrame,
        date_col: str | None = None,
        symbol: str | None = None,
        fill_method: str = "ffill",
    ) -> pd.DataFrame:
        """Standardize columns, sort the index, and handle missing values."""

        frame = data.copy()
        frame = self.standardize_columns(frame)

        inferred_date_col = date_col.lower() if date_col else None
        if inferred_date_col and inferred_date_col not in frame.columns:
            normalized_lookup = {
                _normalize_column_name(column): column for column in frame.columns
            }
            inferred_date_col = normalized_lookup.get(_normalize_column_name(date_col))

        if inferred_date_col and inferred_date_col in frame.columns:
            frame["date"] = pd.to_datetime(frame[inferred_date_col], errors="coerce")
            if inferred_date_col != "date":
                frame = frame.drop(columns=[inferred_date_col], errors="ignore")
        elif "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        elif isinstance(frame.index, pd.DatetimeIndex):
            frame = frame.reset_index().rename(columns={"index": "date"})
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        else:
            raise ValueError("A date column or DatetimeIndex is required for OHLCV data.")

        frame = frame.dropna(subset=["date"])
        frame = frame.sort_values("date")
        frame = frame.drop_duplicates(subset=["date"], keep="last")
        frame = frame.set_index("date")
        if frame.index.tz is not None:
            frame.index = frame.index.tz_localize(None)

        missing_required = [
            column for column in _REQUIRED_PRICE_COLUMNS if column not in frame.columns
        ]
        if missing_required:
            raise ValueError(
                "Missing required OHLC columns: " + ", ".join(sorted(missing_required))
            )

        if "volume" not in frame.columns:
            frame["volume"] = 0.0
        if "adj_close" not in frame.columns:
            frame["adj_close"] = frame["close"]

        numeric_columns = [
            column
            for column in frame.columns
            if column in {
                "open",
                "high",
                "low",
                "close",
                "adj_close",
                "volume",
                "open_interest",
            }
        ]
        for column in numeric_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        frame = frame.dropna(how="all", subset=list(_REQUIRED_PRICE_COLUMNS))
        frame = self._handle_missing_values(frame, fill_method=fill_method)
        if symbol is not None:
            frame["symbol"] = symbol

        ordered_columns = [
            column
            for column in [
                "symbol",
                "open",
                "high",
                "low",
                "close",
                "adj_close",
                "volume",
                "open_interest",
            ]
            if column in frame.columns
        ]
        remainder = [column for column in frame.columns if column not in ordered_columns]
        return frame[ordered_columns + remainder]

    def standardize_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Standardize common market-data column names."""

        rename_map: dict[str, str] = {}
        for column in data.columns:
            normalized = _normalize_column_name(str(column))
            if normalized in _COLUMN_ALIASES:
                rename_map[column] = _COLUMN_ALIASES[normalized]
            else:
                rename_map[column] = str(column).strip().lower().replace(" ", "_")
        return data.rename(columns=rename_map)

    def save_data(self, data: pd.DataFrame, path: PathLike) -> None:
        """Save a standardized data frame to CSV."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(output_path)

    def _handle_missing_values(self, data: pd.DataFrame, fill_method: str) -> pd.DataFrame:
        """Apply a simple, configurable missing-data policy."""

        frame = data.copy()
        price_columns = [
            column
            for column in ("open", "high", "low", "close", "adj_close")
            if column in frame.columns
        ]
        if fill_method == "ffill":
            frame[price_columns] = frame[price_columns].ffill()
        elif fill_method == "bfill":
            frame[price_columns] = frame[price_columns].bfill()
        elif fill_method == "drop":
            frame = frame.dropna(subset=price_columns)
        else:
            raise ValueError(
                "fill_method must be one of {'ffill', 'bfill', 'drop'}."
            )

        if "volume" in frame.columns:
            frame["volume"] = frame["volume"].fillna(0.0)
        if "open_interest" in frame.columns:
            frame["open_interest"] = frame["open_interest"].ffill().fillna(0.0)
        return frame


def align_data_frames(frames: Iterable[pd.DataFrame], how: str = "inner") -> pd.DataFrame:
    """Align multiple data frames on their datetime index."""

    data_frames = [frame.copy() for frame in frames]
    if not data_frames:
        raise ValueError("At least one data frame is required for alignment.")
    aligned = pd.concat(data_frames, axis=1, join=how)
    aligned = aligned.sort_index()
    return aligned
