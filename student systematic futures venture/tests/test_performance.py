"""Tests for performance analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.performance import PerformanceAnalyzer


def test_performance_analyzer_returns_core_metrics() -> None:
    index = pd.date_range("2023-01-01", periods=100, freq="B")
    strategy_returns = pd.Series(np.full(len(index), 0.001), index=index)
    benchmark_returns = pd.Series(np.full(len(index), 0.0007), index=index)

    summary = PerformanceAnalyzer().summarize(strategy_returns, benchmark_returns=benchmark_returns)

    for metric in [
        "cagr",
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "max_drawdown",
        "alpha",
        "beta",
        "information_ratio",
        "win_rate",
        "profit_factor",
        "expectancy",
    ]:
        assert metric in summary.index
