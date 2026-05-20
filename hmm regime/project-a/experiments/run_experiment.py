from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd


os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics.metrics import calculate_performance_metrics
from src.analytics.plotting import (
    generate_full_report,
)
from src.backtester.engine import VectorizedBacktester
from src.backtester.portfolio import RegimeAwarePortfolioAllocator
from src.data.loader import MarketDataLoader
from src.data.preprocess import DataPreprocessor
from src.regime.hmm_regime_detector import HMMRegimeDetector
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.momentum import MomentumStrategy
from src.strategies.pairs_trading import PairsTradingStrategy
from src.utils.helpers import (
    coerce_end_date,
    ensure_directory,
    format_metric_value,
    load_yaml_config,
    save_dataframe,
    set_random_seed,
)


LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure application logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def build_strategies(config: dict[str, dict[str, object]]) -> dict[str, object]:
    """Instantiate all enabled trading strategies from configuration."""
    strategies: dict[str, object] = {}

    momentum_config = config.get("momentum", {})
    if momentum_config.get("enabled", True):
        strategies["momentum"] = MomentumStrategy(
            short_window=int(momentum_config.get("short_window", 50)),
            long_window=int(momentum_config.get("long_window", 200)),
            assets=list(momentum_config.get("assets", [])) or None,
            name="momentum",
        )

    mean_reversion_config = config.get("mean_reversion", {})
    if mean_reversion_config.get("enabled", True):
        strategies["mean_reversion"] = MeanReversionStrategy(
            window=int(mean_reversion_config.get("window", 20)),
            entry_threshold=float(mean_reversion_config.get("entry_threshold", 1.5)),
            exit_threshold=float(mean_reversion_config.get("exit_threshold", 0.5)),
            assets=list(mean_reversion_config.get("assets", [])) or None,
            name="mean_reversion",
        )

    pairs_config = config.get("pairs_trading", {})
    if pairs_config.get("enabled", True):
        strategies["pairs_trading"] = PairsTradingStrategy(
            asset_x=str(pairs_config["asset_x"]),
            asset_y=str(pairs_config["asset_y"]),
            lookback_window=int(pairs_config.get("lookback_window", 63)),
            zscore_window=int(pairs_config.get("zscore_window", 20)),
            entry_threshold=float(pairs_config.get("entry_threshold", 2.0)),
            exit_threshold=float(pairs_config.get("exit_threshold", 0.5)),
            name="pairs_trading",
        )

    if not strategies:
        raise ValueError("No strategies are enabled in the configuration.")

    return strategies


def save_processed_outputs(
    processed_dir: Path,
    clean_prices: pd.DataFrame,
    daily_returns: pd.DataFrame,
    regime_features: pd.DataFrame,
    regime_states: pd.Series,
    regime_labels: pd.Series,
    state_probabilities: pd.DataFrame,
    state_statistics: pd.DataFrame,
) -> None:
    """Save processed datasets used by the notebooks and reports."""
    save_dataframe(clean_prices, processed_dir / "clean_prices.csv")
    save_dataframe(daily_returns, processed_dir / "daily_returns.csv")
    save_dataframe(regime_features, processed_dir / "regime_features.csv")
    save_dataframe(regime_states.to_frame(name="state"), processed_dir / "hmm_states.csv")
    save_dataframe(regime_labels.to_frame(name="regime"), processed_dir / "regime_labels.csv")
    save_dataframe(state_probabilities, processed_dir / "state_probabilities.csv")
    save_dataframe(state_statistics, processed_dir / "state_statistics.csv")


def build_daily_results_frame(
    backtest_result,
    regime_states: pd.Series,
) -> pd.DataFrame:
    """Assemble a flat daily results table."""
    results = pd.concat(
        [
            backtest_result.gross_returns,
            backtest_result.portfolio_returns,
            backtest_result.transaction_costs,
            backtest_result.turnover,
            backtest_result.pnl,
            backtest_result.equity_curve,
            backtest_result.drawdown,
            backtest_result.regime_series,
            regime_states.reindex(backtest_result.equity_curve.index).ffill().rename("state"),
        ],
        axis=1,
    )
    return results


def print_summary(metrics: pd.Series, state_labels: dict[int, str]) -> None:
    """Print a concise run summary to the console."""
    LOGGER.info("Backtest summary")
    for metric_name, metric_value in metrics.items():
        LOGGER.info("  %-22s %s", metric_name + ":", format_metric_value(metric_name, float(metric_value)))

    LOGGER.info("Inferred regime labels")
    for state, label in sorted(state_labels.items()):
        LOGGER.info("  state_%s -> %s", state, label)


def run_pipeline(config_path: Path) -> dict[str, Path]:
    """Execute the full research and backtesting workflow."""
    config = load_yaml_config(config_path)
    set_random_seed(int(config.get("random_seed", 42)))

    data_config = config["data"]
    outputs_config = config["outputs"]
    backtest_config = config["backtest"]

    raw_dir = ensure_directory(PROJECT_ROOT / str(data_config["raw_data_path"]))
    processed_dir = ensure_directory(PROJECT_ROOT / str(data_config["processed_data_path"]))
    figures_dir = ensure_directory(PROJECT_ROOT / str(outputs_config["figures_dir"]))
    reports_dir = ensure_directory(PROJECT_ROOT / str(outputs_config["reports_dir"]))

    loader = MarketDataLoader(raw_data_dir=raw_dir)
    tickers = list(data_config["tickers"])
    benchmark = str(data_config["benchmark"])
    start_date = str(data_config["start_date"])
    end_date = coerce_end_date(str(data_config.get("end_date", "auto")))

    LOGGER.info("Loading market data for %s", ", ".join(tickers))
    prices = loader.load_or_download(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        refresh=bool(data_config.get("refresh_data", False)),
    )

    preprocessor = DataPreprocessor()
    clean_prices = preprocessor.clean_prices(prices)
    daily_returns = preprocessor.compute_returns(clean_prices)
    regime_features = preprocessor.build_regime_features(clean_prices, benchmark=benchmark)

    strategies = build_strategies(config["strategies"])
    strategy_outputs = {
        strategy_name: strategy.generate_positions(clean_prices)
        for strategy_name, strategy in strategies.items()
    }

    hmm_config = config["hmm"]
    regime_detector = HMMRegimeDetector(
        n_regimes=int(hmm_config.get("n_regimes", 3)),
        covariance_type=str(hmm_config.get("covariance_type", "full")),
        n_iter=int(hmm_config.get("n_iter", 250)),
        random_state=int(hmm_config.get("random_state", config.get("random_seed", 42))),
    )
    regime_result = regime_detector.fit_predict(regime_features)

    full_regime_labels = regime_result.labeled_regimes.reindex(clean_prices.index).ffill()
    allocator = RegimeAwarePortfolioAllocator(regime_weight_map=config["allocation"])
    strategy_weights = allocator.build_weight_frame(
        regimes=full_regime_labels,
        strategy_names=list(strategy_outputs.keys()),
    )

    backtester = VectorizedBacktester(transaction_cost_bps=float(backtest_config.get("transaction_cost_bps", 5.0)))
    backtest_result = backtester.run(
        prices=clean_prices,
        strategy_outputs=strategy_outputs,
        strategy_allocations=strategy_weights,
        initial_capital=float(backtest_config.get("initial_capital", 1.0)),
        regimes=full_regime_labels,
    )

    metrics = calculate_performance_metrics(
        returns=backtest_result.portfolio_returns,
        turnover=backtest_result.turnover,
        risk_free_rate=float(backtest_config.get("risk_free_rate", 0.0)),
        periods_per_year=int(backtest_config.get("periods_per_year", 252)),
    )
    metrics_frame = metrics.rename_axis("metric").to_frame(name="value")
    strategy_metrics = pd.DataFrame(
        {
            name: calculate_performance_metrics(
                returns=backtest_result.strategy_returns[name],
                periods_per_year=int(backtest_config.get("periods_per_year", 252)),
            )
            for name in backtest_result.strategy_returns.columns
        }
    ).T

    save_processed_outputs(
        processed_dir=processed_dir,
        clean_prices=clean_prices,
        daily_returns=daily_returns,
        regime_features=regime_features,
        regime_states=regime_result.states,
        regime_labels=regime_result.labeled_regimes,
        state_probabilities=regime_result.state_probabilities,
        state_statistics=regime_result.state_statistics,
    )

    daily_results = build_daily_results_frame(backtest_result=backtest_result, regime_states=regime_result.states)
    save_dataframe(metrics_frame, reports_dir / str(outputs_config["metrics_filename"]))
    save_dataframe(strategy_metrics, reports_dir / str(outputs_config["strategy_metrics_filename"]))
    save_dataframe(daily_results, reports_dir / str(outputs_config["daily_results_filename"]))
    save_dataframe(regime_result.transition_matrix, reports_dir / str(outputs_config["transition_matrix_filename"]))
    save_dataframe(regime_result.state_probabilities, reports_dir / str(outputs_config["state_probabilities_filename"]))
    save_dataframe(strategy_weights, reports_dir / "strategy_weights.csv")
    save_dataframe(backtest_result.strategy_returns, reports_dir / "strategy_daily_returns.csv")

    report_artifacts = generate_full_report(
        results=backtest_result,
        benchmark_returns=daily_returns[benchmark],
        regimes=full_regime_labels,
        strategy_returns=backtest_result.strategy_returns,
        output_dir=figures_dir,
        reports_dir=reports_dir,
        benchmark_prices=clean_prices[benchmark],
        benchmark_name=benchmark,
        prices=clean_prices,
        strategy_outputs=strategy_outputs,
        transition_matrix=regime_result.transition_matrix,
        periods_per_year=int(backtest_config.get("periods_per_year", 252)),
        risk_free_rate=float(backtest_config.get("risk_free_rate", 0.0)),
        random_seed=int(config.get("random_seed", 42)),
    )

    print_summary(metrics=metrics, state_labels=regime_detector.state_labels())

    return {
        "metrics": reports_dir / str(outputs_config["metrics_filename"]),
        "daily_results": reports_dir / str(outputs_config["daily_results_filename"]),
        "figures_dir": figures_dir,
        "dashboard": report_artifacts.reports["interactive_dashboard"],
        "performance_summary": report_artifacts.reports["performance_summary"],
    }


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the regime-aware multi-strategy backtest.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "config.yaml",
        help="Path to the YAML experiment configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    """Program entry point."""
    configure_logging()
    args = parse_args()
    run_pipeline(config_path=args.config.resolve())


if __name__ == "__main__":
    main()
