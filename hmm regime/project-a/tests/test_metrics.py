from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics.metrics import calculate_performance_metrics, drawdown_series
from src.analytics.metrics import conditional_value_at_risk, drawdown_periods, monthly_return_table, value_at_risk


def test_calculate_performance_metrics_returns_expected_fields() -> None:
    returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015], name="returns")
    turnover = pd.Series([0.1, 0.2, 0.05, 0.15, 0.1], name="turnover")

    metrics = calculate_performance_metrics(returns=returns, turnover=turnover)

    expected_cumulative = (1.0 + returns).prod() - 1.0
    assert math.isclose(metrics["Cumulative Return"], expected_cumulative, rel_tol=1e-9)
    assert "Sharpe Ratio" in metrics.index
    assert "Maximum Drawdown" in metrics.index
    assert "Average Turnover" in metrics.index
    assert math.isclose(metrics["Average Turnover"], turnover.mean(), rel_tol=1e-9)


def test_drawdown_series_matches_peak_to_trough_behavior() -> None:
    returns = pd.Series([0.10, -0.05, -0.10, 0.03], name="returns")
    drawdown = drawdown_series(returns)

    assert drawdown.iloc[0] == 0.0
    assert drawdown.min() < 0.0
    assert drawdown.iloc[2] == drawdown.min()


def test_monthly_return_table_contains_annual_total_column() -> None:
    index = pd.date_range("2024-01-01", periods=60, freq="D")
    returns = pd.Series(0.001, index=index)

    table = monthly_return_table(returns)

    assert "Annual Total" in table.columns
    assert len(table.index) == 1


def test_drawdown_periods_and_tail_risk_metrics_are_ordered() -> None:
    index = pd.date_range("2024-01-01", periods=8, freq="D")
    returns = pd.Series([0.02, -0.03, -0.04, 0.01, 0.03, -0.02, 0.01, 0.02], index=index)

    periods = drawdown_periods(returns, top_n=5)
    var_95 = value_at_risk(returns, confidence_level=0.95)
    cvar_95 = conditional_value_at_risk(returns, confidence_level=0.95)

    assert not periods.empty
    assert {"Start Date", "Trough Date", "Recovery Date", "Depth", "Duration"}.issubset(periods.columns)
    assert cvar_95 <= var_95
