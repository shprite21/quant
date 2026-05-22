"""Command-line entry point for the Bayesian statistical arbitrage prototype."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.backtester import BacktestConfig, BacktestResult, BasketBacktester
from src.cointegration import CointegrationTester
from src.data_loader import MarketDataLoader
from src.metrics import MetricsConfig, PerformanceAnalyzer
from src.optimizer import BayesianStrategyOptimizer, OptimizerConfig
from src.signals import SignalEngine
from src.spread_builder import WeightedSpreadBuilder
from src.utils import ResearchPlotter, configure_logging, ensure_directories, load_config, save_json, set_random_seed


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(description="Bayesian Optimization for Cointegration-Based Basket Trading")
    parser.add_argument("--config", default="configs/strategy_config.yaml", help="Path to YAML config relative to project root.")
    parser.add_argument("--use-cached-data", action="store_true", help="Use data/processed/adjusted_close_clean.csv if available.")
    parser.add_argument("--skip-walk-forward", action="store_true", help="Disable walk-forward validation for this run.")
    return parser.parse_args()


def build_optimizer(config: dict[str, Any]) -> BayesianStrategyOptimizer:
    """Construct the optimization stack from config sections."""

    backtest_config = config["backtest"]
    metrics_config = config["metrics"]
    optimizer_config = config["optimizer"]
    spread_config = config["spread"]

    return BayesianStrategyOptimizer(
        config=OptimizerConfig(
            n_trials=int(optimizer_config.get("n_trials", 100)),
            random_seed=int(optimizer_config.get("random_seed", 42)),
            validation_fraction=float(optimizer_config.get("validation_fraction", 0.35)),
            weight_low=float(optimizer_config.get("weight_low", -2.0)),
            weight_high=float(optimizer_config.get("weight_high", 2.0)),
            entry_low=float(optimizer_config.get("entry_low", 1.0)),
            entry_high=float(optimizer_config.get("entry_high", 3.0)),
            exit_low=float(optimizer_config.get("exit_low", 0.05)),
            exit_high=float(optimizer_config.get("exit_high", 1.0)),
            rolling_window_low=int(optimizer_config.get("rolling_window_low", 20)),
            rolling_window_high=int(optimizer_config.get("rolling_window_high", 120)),
            position_size_low=float(optimizer_config.get("position_size_low", 0.25)),
            position_size_high=float(optimizer_config.get("position_size_high", 1.5)),
            min_validation_trades=int(optimizer_config.get("min_validation_trades", 2)),
        ),
        spread_builder=WeightedSpreadBuilder(
            use_log_prices=bool(spread_config.get("use_log_prices", True)),
            shift_statistics=bool(spread_config.get("shift_statistics", True)),
        ),
        signal_engine=SignalEngine(),
        backtester=BasketBacktester(
            BacktestConfig(
                initial_capital=float(backtest_config.get("initial_capital", 1_000_000)),
                transaction_cost_bps=float(backtest_config.get("transaction_cost_bps", 1.0)),
                slippage_bps=float(backtest_config.get("slippage_bps", 1.0)),
                annualization_factor=int(metrics_config.get("annualization_factor", 252)),
            )
        ),
        metrics=PerformanceAnalyzer(
            MetricsConfig(
                annualization_factor=int(metrics_config.get("annualization_factor", 252)),
                risk_free_rate=float(metrics_config.get("risk_free_rate", 0.0)),
            )
        ),
    )


def slice_oos_result(result: BacktestResult, test_index: pd.Index, initial_capital: float) -> BacktestResult:
    """Slice a full warmup-inclusive backtest to the true out-of-sample test period."""

    returns = result.returns.reindex(test_index).fillna(0.0)
    equity = (initial_capital * (1.0 + returns).cumprod()).rename("equity")
    trade_log = result.trade_log.copy()
    if not trade_log.empty:
        trade_log["entry_date"] = pd.to_datetime(trade_log["entry_date"])
        trade_log["exit_date"] = pd.to_datetime(trade_log["exit_date"])
        trade_log = trade_log[
            (trade_log["exit_date"] >= test_index[0]) & (trade_log["entry_date"] <= test_index[-1])
        ]
    return BacktestResult(
        equity_curve=equity,
        returns=returns,
        positions=result.positions.reindex(test_index),
        trade_log=trade_log,
        costs=result.costs.reindex(test_index).fillna(0.0),
        turnover=result.turnover.reindex(test_index).fillna(0.0),
        spread=result.spread.reindex(test_index),
        z_score=result.z_score.reindex(test_index),
        signals=result.signals.reindex(test_index).fillna(0.0),
        weights=result.weights,
    )


def main() -> None:
    """Run the full research pipeline."""

    args = parse_args()
    project_root = Path(__file__).resolve().parent
    config_path = project_root / args.config
    config = load_config(config_path)
    configure_logging(config.get("logging", {}).get("level", "INFO"))
    ensure_directories(project_root)
    set_random_seed(int(config["optimizer"].get("random_seed", 42)))

    LOGGER.info("Loading market data")
    loader = MarketDataLoader.from_config(config, project_root)
    prices = loader.load_or_download(use_cached=args.use_cached_data)
    prices.to_csv(project_root / "data" / "processed" / "adjusted_close_clean.csv")

    LOGGER.info("Running cointegration diagnostics")
    cointegration_config = config["cointegration"]
    tester = CointegrationTester(
        significance_level=float(cointegration_config.get("significance_level", 0.05)),
        use_log_prices=bool(cointegration_config.get("use_log_prices", True)),
        min_obs=int(cointegration_config.get("min_obs", 252)),
    )
    pairwise_results = tester.scan_pairs(prices)
    pairwise_results.to_csv(project_root / "results" / "performance_reports" / "engle_granger_pairs.csv", index=False)

    basket_result = tester.test_basket(prices, target=config["data"]["tickers"][0])
    save_json(
        basket_result.to_dict(),
        project_root / "results" / "performance_reports" / "engle_granger_basket.json",
    )

    if bool(cointegration_config.get("run_johansen", True)):
        johansen_result = tester.johansen_test(
            prices,
            det_order=int(cointegration_config.get("johansen_det_order", 0)),
            k_ar_diff=int(cointegration_config.get("johansen_k_ar_diff", 1)),
        )
        save_json(
            johansen_result.to_dict(),
            project_root / "results" / "performance_reports" / "johansen_basket.json",
        )

    train_fraction = float(config["validation"].get("train_fraction", 0.70))
    split_idx = int(len(prices) * train_fraction)
    train_prices = prices.iloc[:split_idx]
    test_prices = prices.iloc[split_idx:]
    if len(test_prices) < 30:
        raise ValueError("Test split is too small. Increase history length or reduce train_fraction.")

    LOGGER.info("Optimizing strategy on training data")
    optimizer = build_optimizer(config)
    optimization = optimizer.optimize(train_prices)
    optimizer.save_optimization_artifacts(
        optimization.study,
        output_dir=project_root / "results" / "optimization_logs",
        figure_dir=project_root / "results" / "figures",
    )
    optimization.stability.to_csv(project_root / "results" / "optimization_logs" / "parameter_stability.csv")
    save_json(
        {
            "best_value_validation_sharpe": optimization.best_value,
            "best_params": optimization.best_params.to_dict(),
        },
        project_root / "results" / "optimization_logs" / "best_params.json",
    )

    LOGGER.info("Evaluating best parameters out of sample")
    warmup = train_prices.tail(max(optimization.best_params.rolling_window * 2, 30))
    evaluation_prices = pd.concat([warmup, test_prices])
    full_result = optimizer.run_strategy(evaluation_prices, optimization.best_params)
    test_result = slice_oos_result(
        full_result,
        test_prices.index,
        initial_capital=optimizer.backtester.config.initial_capital,
    )
    test_report = optimizer.metrics.calculate(test_result.returns, test_result.equity_curve, test_result.trade_log)
    save_json(test_report, project_root / "results" / "performance_reports" / "out_of_sample_performance.json")

    test_result.trade_log.to_csv(project_root / "results" / "performance_reports" / "trade_log.csv", index=False)
    test_result.positions.to_csv(project_root / "results" / "performance_reports" / "positions.csv")
    pd.concat(
        [
            test_result.equity_curve,
            test_result.returns,
            test_result.costs,
            test_result.turnover,
            test_result.spread,
            test_result.z_score,
            test_result.signals,
        ],
        axis=1,
    ).to_csv(project_root / "results" / "performance_reports" / "equity_curve.csv")

    plotter = ResearchPlotter()
    plotter.plot_spread_and_zscore(
        spread=test_result.spread,
        z_score=test_result.z_score,
        signals=test_result.signals,
        entry_threshold=optimization.best_params.entry_threshold,
        exit_threshold=optimization.best_params.exit_threshold,
        output_path=project_root / "results" / "figures" / "spread_zscore_signals.png",
    )
    plotter.plot_equity_and_drawdown(
        equity_curve=test_result.equity_curve,
        drawdown=optimizer.metrics.drawdown(test_result.equity_curve),
        output_path=project_root / "results" / "figures" / "equity_drawdown.png",
    )

    walk_forward_config = config.get("walk_forward", {})
    if bool(walk_forward_config.get("enabled", False)) and not args.skip_walk_forward:
        LOGGER.info("Running walk-forward validation")
        walk_forward = optimizer.walk_forward_validation(
            prices=prices,
            train_size=int(walk_forward_config.get("train_size", 756)),
            test_size=int(walk_forward_config.get("test_size", 126)),
            step_size=int(walk_forward_config.get("step_size", walk_forward_config.get("test_size", 126))),
            n_trials=int(walk_forward_config.get("n_trials", max(10, optimizer.config.n_trials // 3))),
        )
        walk_forward.to_csv(project_root / "results" / "optimization_logs" / "walk_forward_results.csv", index=False)

    LOGGER.info("Research run complete")
    LOGGER.info("Out-of-sample Sharpe: %.3f", test_report["sharpe_ratio"])
    LOGGER.info("Out-of-sample CAGR: %.3f", test_report["cagr"])
    LOGGER.info("Out-of-sample max drawdown: %.3f", test_report["max_drawdown"])


if __name__ == "__main__":
    main()

