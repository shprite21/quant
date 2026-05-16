from __future__ import annotations

import numpy as np
import pandas as pd


def _clean_returns(returns: pd.Series) -> pd.Series:
    series = pd.Series(returns, copy=False).replace([np.inf, -np.inf], np.nan).dropna()
    return series.astype(float)


def cumulative_return(returns: pd.Series) -> float:
    series = _clean_returns(returns)
    if series.empty:
        return float("nan")
    return float((1 + series).prod() - 1)


def annual_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    series = _clean_returns(returns)
    if series.empty:
        return float("nan")

    years = len(series) / periods_per_year
    if years <= 0:
        return float("nan")

    compounded = (1 + series).prod()
    return float(compounded ** (1 / years) - 1)


def annual_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    series = _clean_returns(returns)
    if series.empty:
        return float("nan")
    return float(series.std(ddof=0) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    series = _clean_returns(returns)
    if series.empty:
        return float("nan")

    excess_returns = series - risk_free_rate / periods_per_year
    volatility = excess_returns.std(ddof=0)
    if volatility == 0:
        return float("nan")
    return float(excess_returns.mean() / volatility * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    series = _clean_returns(returns)
    if series.empty:
        return float("nan")

    excess_returns = series - risk_free_rate / periods_per_year
    downside = excess_returns.clip(upper=0)
    downside_volatility = downside.std(ddof=0)
    if downside_volatility == 0:
        return float("nan")
    return float(excess_returns.mean() / downside_volatility * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: pd.Series) -> float:
    series = pd.Series(equity_curve, copy=False).replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if series.empty:
        return float("nan")

    drawdown = series / series.cummax() - 1
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series, equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    cagr = annual_return(returns, periods_per_year=periods_per_year)
    worst_drawdown = abs(max_drawdown(equity_curve))
    if worst_drawdown == 0 or np.isnan(worst_drawdown):
        return float("nan")
    return float(cagr / worst_drawdown)


def hit_rate(returns: pd.Series) -> float:
    series = _clean_returns(returns)
    if series.empty:
        return float("nan")

    non_zero = series[series != 0]
    if non_zero.empty:
        return float("nan")
    return float((non_zero > 0).mean())


def profit_factor(returns: pd.Series) -> float:
    series = _clean_returns(returns)
    gains = series[series > 0].sum()
    losses = series[series < 0].sum()
    if losses == 0:
        return float("nan")
    return float(gains / abs(losses))


def summarize_performance(
    returns: pd.Series,
    equity_curve: pd.Series | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> pd.Series:
    """Compute common portfolio performance statistics."""

    series = _clean_returns(returns)
    if equity_curve is None:
        equity_curve = (1 + series).cumprod()

    summary = pd.Series(
        {
            "cumulative_return": cumulative_return(series),
            "annual_return": annual_return(series, periods_per_year=periods_per_year),
            "annual_volatility": annual_volatility(series, periods_per_year=periods_per_year),
            "sharpe_ratio": sharpe_ratio(series, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year),
            "sortino_ratio": sortino_ratio(series, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year),
            "max_drawdown": max_drawdown(equity_curve),
            "calmar_ratio": calmar_ratio(series, equity_curve, periods_per_year=periods_per_year),
            "hit_rate": hit_rate(series),
            "profit_factor": profit_factor(series),
        }
    )
    return summary
