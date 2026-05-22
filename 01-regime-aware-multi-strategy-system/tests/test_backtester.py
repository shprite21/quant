from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtester.engine import VectorizedBacktester
from src.backtester.portfolio import RegimeAwarePortfolioAllocator
from src.strategies.base import StrategyOutput


def test_backtester_shifts_positions_to_avoid_lookahead_bias() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    prices = pd.DataFrame({"SPY": [100.0, 110.0, 121.0]}, index=index)
    positions = pd.DataFrame({"SPY": [0.0, 1.0, 1.0]}, index=index)
    strategy_output = StrategyOutput(name="momentum", positions=positions)

    regimes = pd.Series(["Bull", "Bull", "Bull"], index=index, name="regime")
    allocator = RegimeAwarePortfolioAllocator(
        regime_weight_map={"Bull": {"momentum": 1.0}},
    )
    strategy_weights = allocator.build_weight_frame(regimes=regimes, strategy_names=["momentum"])

    backtester = VectorizedBacktester(transaction_cost_bps=10.0)
    result = backtester.run(
        prices=prices,
        strategy_outputs={"momentum": strategy_output},
        strategy_allocations=strategy_weights,
        initial_capital=1.0,
        regimes=regimes,
    )

    assert result.portfolio_returns.iloc[1] == 0.0
    assert math.isclose(result.turnover.iloc[2], 1.0, rel_tol=1e-9)
    assert math.isclose(result.portfolio_returns.iloc[2], 0.099, rel_tol=1e-9)
