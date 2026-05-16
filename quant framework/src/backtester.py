from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    """Simple configuration object for a vectorized directional backtest."""

    initial_capital: float = 100_000.0
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 2.0
    max_leverage: float = 1.0
    position_delay: int = 1
    price_col: str = "close"


def backtest_signals(
    price_frame: pd.DataFrame,
    signals: pd.Series,
    config: BacktestConfig | None = None,
) -> pd.DataFrame:
    """Backtest long, flat, or short signals on a single close-price series."""

    config = config or BacktestConfig()

    if config.price_col not in price_frame.columns:
        raise ValueError(f"'{config.price_col}' was not found in the input DataFrame.")

    prices = price_frame[config.price_col].astype(float)
    asset_returns = prices.pct_change().fillna(0.0)

    raw_signal = signals.reindex(price_frame.index).fillna(0.0).clip(
        lower=-config.max_leverage,
        upper=config.max_leverage,
    )
    position = raw_signal.shift(config.position_delay).fillna(0.0)
    turnover = position.diff().abs().fillna(position.abs())

    total_cost_rate = (config.transaction_cost_bps + config.slippage_bps) / 10_000
    gross_return = position * asset_returns
    trading_cost = turnover * total_cost_rate
    strategy_return = gross_return - trading_cost
    equity_curve = config.initial_capital * (1 + strategy_return).cumprod()

    backtest_frame = pd.DataFrame(
        {
            "price": prices,
            "asset_return": asset_returns,
            "signal": raw_signal,
            "position": position,
            "turnover": turnover,
            "gross_return": gross_return,
            "trading_cost": trading_cost,
            "strategy_return": strategy_return,
            "equity_curve": equity_curve,
        },
        index=price_frame.index,
    )
    backtest_frame["drawdown"] = backtest_frame["equity_curve"] / backtest_frame["equity_curve"].cummax() - 1
    return backtest_frame


def extract_trade_points(backtest_frame: pd.DataFrame) -> pd.DataFrame:
    """Return timestamps where the held position changes."""

    if "position" not in backtest_frame.columns:
        raise ValueError("The input frame must contain a 'position' column.")

    changes = backtest_frame["position"].diff().fillna(backtest_frame["position"])
    trades = backtest_frame.loc[changes != 0, ["price", "signal", "position"]].copy()
    trades["position_change"] = changes.loc[trades.index]
    return trades

