"""Feature engineering utilities for regime and signal models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


def _validate_ohlcv_columns(data: pd.DataFrame) -> None:
    required = {"open", "high", "low", "close", "volume"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Missing OHLCV columns required for features: {missing}")


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _true_range(data: pd.DataFrame) -> pd.Series:
    close = data["close"]
    ranges = pd.concat(
        [
            data["high"] - data["low"],
            (data["high"] - close.shift(1)).abs(),
            (data["low"] - close.shift(1)).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Compute the Relative Strength Index."""

    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    average_gain = gains.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    average_loss = losses.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    relative_strength = average_gain / average_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + relative_strength))


def compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Compute MACD, signal line, and histogram."""

    fast_ema = _ema(close, span=fast)
    slow_ema = _ema(close, span=slow)
    macd = fast_ema - slow_ema
    signal_line = _ema(macd, span=signal)
    histogram = macd - signal_line
    return pd.DataFrame(
        {
            "macd": macd,
            "macd_signal": signal_line,
            "macd_hist": histogram,
        },
        index=close.index,
    )


def compute_atr(data: pd.DataFrame, window: int = 14) -> pd.Series:
    """Compute Average True Range."""

    return _true_range(data).rolling(window=window, min_periods=window).mean()


def compute_bollinger_bandwidth(
    close: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """Compute Bollinger band levels and normalized width."""

    rolling_mean = close.rolling(window=window, min_periods=window).mean()
    rolling_std = close.rolling(window=window, min_periods=window).std(ddof=0)
    upper = rolling_mean + num_std * rolling_std
    lower = rolling_mean - num_std * rolling_std
    width = (upper - lower) / rolling_mean.replace(0.0, np.nan)
    return pd.DataFrame(
        {
            "bollinger_mid": rolling_mean,
            "bollinger_upper": upper,
            "bollinger_lower": lower,
            "bollinger_bandwidth": width,
        },
        index=close.index,
    )


def compute_adx(data: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Compute ADX and directional movement indicators."""

    high_diff = data["high"].diff()
    low_diff = -data["low"].diff()

    plus_dm = pd.Series(
        np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0),
        index=data.index,
    )
    minus_dm = pd.Series(
        np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0),
        index=data.index,
    )

    true_range = _true_range(data)
    atr = true_range.rolling(window=window, min_periods=window).mean()
    plus_di = 100.0 * (
        plus_dm.rolling(window=window, min_periods=window).sum()
        / atr.replace(0.0, np.nan)
    )
    minus_di = 100.0 * (
        minus_dm.rolling(window=window, min_periods=window).sum()
        / atr.replace(0.0, np.nan)
    )
    dx = (
        100.0
        * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0.0, np.nan)
    )
    adx = dx.rolling(window=window, min_periods=window).mean()
    return pd.DataFrame(
        {
            "plus_di": plus_di,
            "minus_di": minus_di,
            "adx": adx,
        },
        index=data.index,
    )


def _moving_average_slope(close: pd.Series, ma_window: int, slope_window: int = 5) -> pd.Series:
    moving_average = close.rolling(window=ma_window, min_periods=ma_window).mean()
    return moving_average.pct_change(periods=slope_window)


@dataclass(slots=True)
class FeatureEngineer:
    """Build research and trading features from OHLCV data."""

    annualization_factor: int = 252
    lag_features_by: int = 1

    def build_features(
        self,
        data: pd.DataFrame,
        india_vix: pd.Series | pd.DataFrame | None = None,
        derivatives_data: pd.DataFrame | None = None,
        dropna: bool = False,
    ) -> pd.DataFrame:
        """Create a feature-rich data frame without forward-looking leakage."""

        frame = data.copy()
        _validate_ohlcv_columns(frame)

        close = frame["close"]
        log_close = np.log(close.replace(0.0, np.nan))
        daily_return = close.pct_change()
        log_return = log_close.diff()

        frame["daily_return"] = daily_return
        frame["log_return"] = log_return
        frame["drawdown"] = close / close.cummax() - 1.0

        for window in (5, 10, 20):
            frame[f"momentum_{window}"] = close.pct_change(window)
            frame[f"rolling_volatility_{window}"] = (
                daily_return.rolling(window=window, min_periods=window).std(ddof=0)
                * np.sqrt(self.annualization_factor)
            )
            frame[f"realized_volatility_{window}"] = (
                log_return.rolling(window=window, min_periods=window).std(ddof=0)
                * np.sqrt(self.annualization_factor)
            )
            frame[f"ma_slope_{window}"] = _moving_average_slope(
                close=close,
                ma_window=window,
                slope_window=max(2, window // 2),
            )

        frame["rsi_14"] = compute_rsi(close, window=14)

        macd_frame = compute_macd(close)
        frame = frame.join(macd_frame)

        atr = compute_atr(frame, window=14)
        frame["atr_14"] = atr
        frame["atr_percent"] = atr / close.replace(0.0, np.nan)

        bollinger_frame = compute_bollinger_bandwidth(close, window=20, num_std=2.0)
        frame = frame.join(bollinger_frame)

        adx_frame = compute_adx(frame, window=14)
        frame = frame.join(adx_frame)

        frame["volume_change"] = frame["volume"].pct_change()
        frame["relative_volume_20"] = frame["volume"] / (
            frame["volume"].rolling(window=20, min_periods=20).mean()
        )

        frame["trend_strength"] = (
            frame["adx"].fillna(0.0)
            * frame["ma_slope_20"].abs().fillna(0.0)
        )

        if india_vix is not None:
            frame = self._add_india_vix(frame, india_vix)

        if derivatives_data is not None:
            frame = self._add_derivatives_features(frame, derivatives_data)
        else:
            frame = self._attach_existing_derivatives_columns(frame)

        engineered_columns = self.get_feature_columns(frame, original_columns=data.columns)
        if self.lag_features_by > 0:
            frame[engineered_columns] = frame[engineered_columns].shift(self.lag_features_by)

        if dropna:
            frame = frame.dropna(subset=engineered_columns)
        return frame

    def add_regime_features(
        self,
        features: pd.DataFrame,
        regime_output: pd.DataFrame,
        lag_regime_by: int | None = None,
    ) -> pd.DataFrame:
        """Merge regime labels, probabilities, and durations into a feature table."""

        frame = features.copy()
        regime_frame = regime_output.copy()
        common_columns = [
            column
            for column in regime_frame.columns
            if column.startswith("regime_") or column in {"state_probability", "regime_duration"}
        ]
        if not common_columns:
            raise ValueError("regime_output must contain regime-related columns.")
        merged = frame.join(regime_frame[common_columns], how="left")
        shift_amount = self.lag_features_by if lag_regime_by is None else lag_regime_by
        if shift_amount > 0:
            merged[common_columns] = merged[common_columns].shift(shift_amount)
        return merged

    def get_feature_columns(
        self,
        data: pd.DataFrame,
        original_columns: Iterable[str] | None = None,
        extra_exclusions: Iterable[str] | None = None,
    ) -> list[str]:
        """Return engineered feature columns suitable for modeling."""

        exclusions = set(original_columns or [])
        exclusions.update(
            {
                "target",
                "target_class",
                "target_regression",
                "forward_return",
                "strategy",
                "position",
            }
        )
        if extra_exclusions is not None:
            exclusions.update(extra_exclusions)
        return [column for column in data.columns if column not in exclusions]

    def _add_india_vix(
        self,
        data: pd.DataFrame,
        india_vix: pd.Series | pd.DataFrame,
    ) -> pd.DataFrame:
        frame = data.copy()
        if isinstance(india_vix, pd.Series):
            vix_series = india_vix.rename("india_vix")
        else:
            vix_frame = india_vix.copy()
            lower_map = {column.lower(): column for column in vix_frame.columns}
            candidate = lower_map.get("close") or lower_map.get("india_vix") or next(
                iter(vix_frame.columns)
            )
            vix_series = vix_frame[candidate].rename("india_vix")
        frame = frame.join(vix_series, how="left")
        frame["india_vix"] = frame["india_vix"].ffill()
        return frame

    def _add_derivatives_features(
        self,
        data: pd.DataFrame,
        derivatives_data: pd.DataFrame,
    ) -> pd.DataFrame:
        frame = data.copy()
        derivatives_frame = derivatives_data.copy()
        derivatives_frame.columns = [
            str(column).strip().lower().replace(" ", "_")
            for column in derivatives_frame.columns
        ]
        keep_columns = [
            column
            for column in derivatives_frame.columns
            if column
            in {
                "open_interest",
                "put_call_ratio",
                "implied_volatility",
                "iv_skew",
                "term_structure",
            }
        ]
        if not keep_columns:
            return frame
        derivatives_frame = derivatives_frame[keep_columns]
        merged = frame.join(derivatives_frame, how="left")
        return self._attach_existing_derivatives_columns(merged)

    def _attach_existing_derivatives_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        frame = data.copy()
        if "open_interest" in frame.columns:
            frame["open_interest_change"] = frame["open_interest"].pct_change()
        if "put_call_ratio" in frame.columns:
            frame["put_call_ratio_zscore_20"] = self._rolling_zscore(
                frame["put_call_ratio"], window=20
            )
        if "implied_volatility" in frame.columns:
            frame["iv_rank_60"] = self._rolling_min_max_rank(
                frame["implied_volatility"], window=60
            )
        if "iv_skew" in frame.columns:
            frame["iv_skew_zscore_20"] = self._rolling_zscore(frame["iv_skew"], window=20)
        if "term_structure" in frame.columns:
            frame["term_structure_change"] = frame["term_structure"].diff()
        return frame

    @staticmethod
    def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
        rolling_mean = series.rolling(window=window, min_periods=window).mean()
        rolling_std = series.rolling(window=window, min_periods=window).std(ddof=0)
        return (series - rolling_mean) / rolling_std.replace(0.0, np.nan)

    @staticmethod
    def _rolling_min_max_rank(series: pd.Series, window: int) -> pd.Series:
        rolling_min = series.rolling(window=window, min_periods=window).min()
        rolling_max = series.rolling(window=window, min_periods=window).max()
        denominator = (rolling_max - rolling_min).replace(0.0, np.nan)
        return (series - rolling_min) / denominator
