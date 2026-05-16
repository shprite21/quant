from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    """Compute a rolling z-score for a single pandas Series."""

    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std()
    return (series - rolling_mean) / rolling_std.replace(0, np.nan)


def relative_strength_index(close: pd.Series, window: int = 14) -> pd.Series:
    """Calculate the RSI technical indicator."""

    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    average_gain = gains.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    average_loss = losses.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    relative_strength = average_gain / average_loss.replace(0, np.nan)
    return 100 - (100 / (1 + relative_strength))


def average_true_range(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    """Calculate average true range."""

    true_range = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window=window, min_periods=window).mean()


def moving_average_convergence_divergence(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Calculate MACD line, signal line, and histogram."""

    fast_ema = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    slow_ema = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd_line - signal_line

    return pd.DataFrame(
        {
            "macd_line": macd_line,
            "macd_signal": signal_line,
            "macd_histogram": histogram,
        },
        index=close.index,
    )


def bollinger_band_width(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    """Calculate Bollinger Band width normalized by the rolling mean."""

    rolling_mean = close.rolling(window=window, min_periods=window).mean()
    rolling_std = close.rolling(window=window, min_periods=window).std()
    upper_band = rolling_mean + num_std * rolling_std
    lower_band = rolling_mean - num_std * rolling_std
    return (upper_band - lower_band) / rolling_mean.replace(0, np.nan)


def rolling_drawdown(close: pd.Series, window: int = 63) -> pd.Series:
    """Measure drawdown from the rolling peak over a lookback window."""

    rolling_peak = close.rolling(window=window, min_periods=window).max()
    return close / rolling_peak - 1


def build_technical_features(
    frame: pd.DataFrame,
    feature_windows: tuple[int, ...] = (5, 10, 20, 63),
    price_col: str = "close",
    volume_col: str = "volume",
) -> pd.DataFrame:
    """Create a technical feature matrix for regime and signal models."""

    if price_col not in frame.columns:
        raise ValueError(f"'{price_col}' was not found in the input DataFrame.")

    close = frame[price_col].astype(float)
    features = pd.DataFrame(index=frame.index)
    log_returns = np.log(close).diff()

    features["log_return_1d"] = log_returns
    features["forward_vol_proxy_5d"] = log_returns.rolling(5, min_periods=5).std() * np.sqrt(252)
    features["rsi_14"] = relative_strength_index(close, window=14)
    features["price_zscore_20d"] = rolling_zscore(close, window=20)
    features["bollinger_width_20d"] = bollinger_band_width(close, window=20)
    features["downside_vol_20d"] = log_returns.clip(upper=0).rolling(20, min_periods=20).std() * np.sqrt(252)
    features["drawdown_63d"] = rolling_drawdown(close, window=63)
    features = features.join(moving_average_convergence_divergence(close))

    for window in feature_windows:
        rolling_mean = close.rolling(window=window, min_periods=window).mean()
        features[f"return_{window}d"] = close.pct_change(window)
        features[f"momentum_{window}d"] = close / close.shift(window) - 1
        features[f"realized_vol_{window}d"] = log_returns.rolling(window, min_periods=window).std() * np.sqrt(252)
        features[f"sma_gap_{window}d"] = close / rolling_mean - 1

    if {"high", "low", price_col}.issubset(frame.columns):
        atr = average_true_range(frame["high"], frame["low"], close, window=14)
        features["atr_pct_14"] = atr / close
        features["range_pct_20d"] = (
            frame["high"].rolling(20, min_periods=20).max()
            - frame["low"].rolling(20, min_periods=20).min()
        ) / close

    if volume_col in frame.columns:
        volume = frame[volume_col].astype(float)
        features["volume_change_1d"] = volume.pct_change()
        features["volume_zscore_20d"] = rolling_zscore(np.log1p(volume), window=20)

    if "open_interest" in frame.columns:
        open_interest = frame["open_interest"].astype(float)
        features["open_interest_change_1d"] = open_interest.pct_change()
        features["open_interest_zscore_20d"] = rolling_zscore(np.log1p(open_interest), window=20)

    if "put_call_ratio" in frame.columns:
        put_call_ratio = frame["put_call_ratio"].astype(float)
        features["put_call_ratio"] = put_call_ratio
        features["put_call_ratio_zscore_20d"] = rolling_zscore(put_call_ratio, window=20)

    for vix_column in ("india_vix", "vix", "implied_volatility_index"):
        if vix_column in frame.columns:
            vix_series = frame[vix_column].astype(float)
            features[f"{vix_column}_level"] = vix_series
            features[f"{vix_column}_change_5d"] = vix_series.pct_change(5)
            features[f"{vix_column}_zscore_20d"] = rolling_zscore(vix_series, window=20)

    features["autocorr_20d"] = log_returns.rolling(20, min_periods=20).apply(
        lambda values: pd.Series(values).autocorr(lag=1),
        raw=False,
    )

    return features.replace([np.inf, -np.inf], np.nan)


def clean_feature_matrix(feature_frame: pd.DataFrame, min_non_null_fraction: float = 0.8) -> pd.DataFrame:
    """Drop sparse columns and rows that are entirely missing."""

    if not 0 < min_non_null_fraction <= 1:
        raise ValueError("min_non_null_fraction must be in the interval (0, 1].")

    required_count = int(len(feature_frame) * min_non_null_fraction)
    cleaned = feature_frame.dropna(axis=1, thresh=required_count)
    return cleaned.dropna(how="all")
