"""Institutional-style performance analytics for trading strategies."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


def equity_curve(returns: pd.Series, initial_capital: float = 1.0) -> pd.Series:
    """Compute an equity curve from periodic returns."""

    clean_returns = returns.fillna(0.0)
    return initial_capital * (1.0 + clean_returns).cumprod()


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Compute the running drawdown series."""

    curve = equity_curve(returns)
    peak = curve.cummax()
    return curve / peak - 1.0


def cagr(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Compute compounded annual growth rate."""

    clean_returns = returns.dropna()
    if clean_returns.empty:
        return 0.0
    total_return = (1.0 + clean_returns).prod()
    years = len(clean_returns) / periods_per_year
    if years <= 0 or total_return <= 0:
        return 0.0
    return float(total_return ** (1.0 / years) - 1.0)


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Compute the annualized Sharpe ratio."""

    excess_returns = returns.dropna() - risk_free_rate / periods_per_year
    volatility = excess_returns.std(ddof=0)
    if volatility == 0 or np.isnan(volatility):
        return 0.0
    return float(np.sqrt(periods_per_year) * excess_returns.mean() / volatility)


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Compute the annualized Sortino ratio."""

    excess_returns = returns.dropna() - risk_free_rate / periods_per_year
    downside = excess_returns[excess_returns < 0].std(ddof=0)
    if downside == 0 or np.isnan(downside):
        return 0.0
    return float(np.sqrt(periods_per_year) * excess_returns.mean() / downside)


def max_drawdown(returns: pd.Series) -> float:
    """Compute maximum drawdown."""

    drawdown = drawdown_series(returns)
    return float(drawdown.min()) if not drawdown.empty else 0.0


def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Compute the Calmar ratio."""

    maximum_drawdown = abs(max_drawdown(returns))
    if maximum_drawdown == 0:
        return 0.0
    return float(cagr(returns, periods_per_year=periods_per_year) / maximum_drawdown)


def alpha_beta(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> tuple[float, float]:
    """Estimate annualized alpha and beta relative to a benchmark."""

    aligned = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
    if aligned.empty:
        return 0.0, 0.0

    strategy_excess = aligned.iloc[:, 0] - risk_free_rate / periods_per_year
    benchmark_excess = aligned.iloc[:, 1] - risk_free_rate / periods_per_year
    benchmark_with_constant = sm.add_constant(benchmark_excess)
    model = sm.OLS(strategy_excess, benchmark_with_constant).fit()
    annualized_alpha = float(model.params["const"] * periods_per_year)
    beta = float(model.params.iloc[1])
    return annualized_alpha, beta


def information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Compute the annualized information ratio."""

    active_returns = (returns - benchmark_returns).dropna()
    tracking_error = active_returns.std(ddof=0)
    if tracking_error == 0 or np.isnan(tracking_error):
        return 0.0
    return float(np.sqrt(periods_per_year) * active_returns.mean() / tracking_error)


def win_rate(returns: pd.Series) -> float:
    """Compute the fraction of positive return observations."""

    trades = returns[returns != 0].dropna()
    if trades.empty:
        return 0.0
    return float((trades > 0).mean())


def profit_factor(returns: pd.Series) -> float:
    """Compute profit factor as gross profits divided by gross losses."""

    trades = returns[returns != 0].dropna()
    gross_profit = trades[trades > 0].sum()
    gross_loss = abs(trades[trades < 0].sum())
    if gross_loss == 0:
        return 0.0
    return float(gross_profit / gross_loss)


def expectancy(returns: pd.Series) -> float:
    """Compute average return per non-zero observation."""

    trades = returns[returns != 0].dropna()
    if trades.empty:
        return 0.0
    return float(trades.mean())


def rolling_sharpe(
    returns: pd.Series,
    window: int = 63,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> pd.Series:
    """Compute a rolling Sharpe ratio."""

    excess = returns.fillna(0.0) - risk_free_rate / periods_per_year
    mean = excess.rolling(window=window, min_periods=window).mean()
    std = excess.rolling(window=window, min_periods=window).std(ddof=0)
    sharpe = np.sqrt(periods_per_year) * mean / std.replace(0.0, np.nan)
    return sharpe


class PerformanceAnalyzer:
    """Create metric summaries and plots from strategy return series."""

    def __init__(self, periods_per_year: int = 252, risk_free_rate: float = 0.0) -> None:
        self.periods_per_year = periods_per_year
        self.risk_free_rate = risk_free_rate

    def summarize(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series | None = None,
    ) -> pd.Series:
        """Build a full performance metric summary."""

        summary = {
            "cagr": cagr(returns, periods_per_year=self.periods_per_year),
            "sharpe_ratio": sharpe_ratio(
                returns,
                risk_free_rate=self.risk_free_rate,
                periods_per_year=self.periods_per_year,
            ),
            "sortino_ratio": sortino_ratio(
                returns,
                risk_free_rate=self.risk_free_rate,
                periods_per_year=self.periods_per_year,
            ),
            "calmar_ratio": calmar_ratio(returns, periods_per_year=self.periods_per_year),
            "max_drawdown": max_drawdown(returns),
            "win_rate": win_rate(returns),
            "profit_factor": profit_factor(returns),
            "expectancy": expectancy(returns),
        }
        if benchmark_returns is not None:
            alpha, beta = alpha_beta(
                returns,
                benchmark_returns,
                risk_free_rate=self.risk_free_rate,
                periods_per_year=self.periods_per_year,
            )
            summary["alpha"] = alpha
            summary["beta"] = beta
            summary["information_ratio"] = information_ratio(
                returns,
                benchmark_returns,
                periods_per_year=self.periods_per_year,
            )
        else:
            summary["alpha"] = 0.0
            summary["beta"] = 0.0
            summary["information_ratio"] = 0.0
        return pd.Series(summary)

    def plot_equity_curve(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series | None = None,
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        """Plot the strategy equity curve and optional benchmark."""

        figure, axis = plt.subplots(figsize=(12, 6))
        equity_curve(returns).plot(ax=axis, label="Strategy", linewidth=2)
        if benchmark_returns is not None:
            equity_curve(benchmark_returns).plot(ax=axis, label="Benchmark", linewidth=1.5)
        axis.set_title("Equity Curve")
        axis.set_ylabel("Growth of 1 Unit")
        axis.legend()
        axis.grid(True, alpha=0.3)
        if save_path is not None:
            figure.savefig(save_path, dpi=150, bbox_inches="tight")
        return figure

    def plot_drawdown(
        self,
        returns: pd.Series,
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        """Plot drawdown through time."""

        figure, axis = plt.subplots(figsize=(12, 4))
        drawdown_series(returns).plot(ax=axis, color="firebrick", linewidth=1.5)
        axis.set_title("Drawdown")
        axis.set_ylabel("Drawdown")
        axis.grid(True, alpha=0.3)
        if save_path is not None:
            figure.savefig(save_path, dpi=150, bbox_inches="tight")
        return figure

    def plot_rolling_sharpe(
        self,
        returns: pd.Series,
        window: int = 63,
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        """Plot rolling Sharpe ratio."""

        figure, axis = plt.subplots(figsize=(12, 4))
        rolling_sharpe(
            returns,
            window=window,
            risk_free_rate=self.risk_free_rate,
            periods_per_year=self.periods_per_year,
        ).plot(ax=axis, color="darkgreen", linewidth=1.5)
        axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        axis.set_title(f"Rolling Sharpe Ratio ({window} periods)")
        axis.grid(True, alpha=0.3)
        if save_path is not None:
            figure.savefig(save_path, dpi=150, bbox_inches="tight")
        return figure
