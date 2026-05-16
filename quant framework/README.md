# Regime-Aware Machine Learning System for Swing Trading and Options Strategy Selection Using Hidden Markov Models and XGBoost

This repository is a starter quantitative research framework for a regime-aware swing-trading and derivatives project. It combines:

1. Unsupervised learning for hidden market-state detection.
2. Supervised learning for short-horizon directional prediction.
3. Rule-based options strategy selection.
4. Walk-forward evaluation with trading performance metrics.

The structure is designed to be portfolio-ready while still being easy to extend into a more serious research stack.

## Repository Layout

```text
regime-aware-derivatives-trading/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_regime_detection.ipynb
│   ├── 04_signal_generation.ipynb
│   ├── 05_options_strategy_selection.ipynb
│   └── 06_backtesting.ipynb
├── results/
├── src/
│   ├── __init__.py
│   ├── backtester.py
│   ├── data_loader.py
│   ├── features.py
│   ├── options_strategy.py
│   ├── performance.py
│   ├── regime_detection.py
│   ├── signal_model.py
│   └── utils.py
├── requirements.txt
└── README.md
```

## Recommended Model Stack

| Component | Primary Choice | Baseline or Benchmarks |
| --- | --- | --- |
| Regime detection | Hidden Markov Model | Gaussian Mixture Model |
| Signal prediction | XGBoost | Logistic Regression, Random Forest |
| Strategy mapping | Rule-based | Rule variants by IV rank or VIX regime |
| Evaluation | Walk-forward backtest | Buy-and-hold and simple trend-following baselines |

## What This Starter System Does

- Loads OHLCV data from local CSV files or `yfinance`.
- Builds technical, volatility, drawdown, volume, and optional derivatives-aware features.
- Fits a Hidden Markov Model to infer latent market regimes.
- Preserves a GMM regime baseline for comparison.
- Trains an XGBoost-based signal model with Logistic Regression and Random Forest alternatives.
- Produces walk-forward out-of-sample predictions and directional signals.
- Maps regime plus signal into options strategy suggestions.
- Backtests signals with transaction costs and reports risk-adjusted metrics.

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If you want to use the repo as a package from the project root:

```bash
set PYTHONPATH=src
```

## Research Pipeline

```text
Market Data
   ↓
Feature Engineering
   ↓
HMM Regime Detection
   ↓
XGBoost Signal Prediction
   ↓
Probability-to-Signal Mapping
   ↓
Options Strategy Selection
   ↓
Walk-Forward Backtesting
   ↓
Performance Evaluation
```

## Suggested Workflow

1. Use `notebooks/01_data_collection.ipynb` to import or download daily market data.
2. Use `notebooks/02_feature_engineering.ipynb` to build the feature set.
3. Use `notebooks/03_regime_detection.ipynb` to fit the HMM and inspect state summaries.
4. Use `notebooks/04_signal_generation.ipynb` to generate walk-forward predictions with XGBoost and benchmark models.
5. Use `notebooks/05_options_strategy_selection.ipynb` to map signals and regimes into options strategies.
6. Use `notebooks/06_backtesting.ipynb` to evaluate trading performance.

## Example End-to-End Flow

```python
from data_loader import load_ohlcv_csv
from features import build_technical_features
from regime_detection import HiddenMarkovRegimeDetector
from signal_model import (
    create_forward_return_targets,
    evaluate_classification_predictions,
    walk_forward_train_predict,
)
from backtester import BacktestConfig, backtest_signals
from performance import summarize_performance

prices = load_ohlcv_csv("data/raw/SPY.csv")
feature_frame = build_technical_features(prices)

regime_model = HiddenMarkovRegimeDetector(n_regimes=4, random_state=42)
regimes = regime_model.fit_predict(feature_frame)

model_input = feature_frame.join(regimes)
targets = create_forward_return_targets(
    prices,
    horizon=5,
    positive_threshold=0.02,
    label_mode="binary",
)

predictions = walk_forward_train_predict(
    model_input,
    targets["target"],
    model_name="xgboost",
    min_train_size=504,
    test_window=21,
)

classification_metrics = evaluate_classification_predictions(
    predictions["target"],
    predictions["prediction"],
    predictions.filter(like="class_"),
)

backtest = backtest_signals(prices, predictions["signal"], BacktestConfig())
trading_metrics = summarize_performance(backtest["strategy_return"], backtest["equity_curve"])

print(classification_metrics)
print(trading_metrics)
```

## Feature Ideas

The current feature builder already includes many of these:

- Daily returns and rolling volatility.
- Momentum over 5, 10, 20, and 63 days.
- Drawdown and price z-score.
- RSI, MACD, ATR, and Bollinger Band width.
- Volume change and volume z-score.
- Optional open interest, put-call ratio, and VIX-style columns if present in the data.
- Regime state assignments and state probabilities from the HMM.

## Evaluation

Classification metrics:

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC

Trading metrics:

- Cumulative return
- Annual return
- Sharpe ratio
- Sortino ratio
- Maximum drawdown
- Calmar ratio
- Hit rate
- Profit factor

## Important Extensions

- Add continuous futures construction and roll logic.
- Add implied-volatility term structure, skew, and realized versus implied spread features.
- Compare HMM states against GMM clusters more formally.
- Add benchmark signals such as moving-average crossover or buy-and-hold.
- Save confusion matrices, state transition plots, equity curves, and feature-importance charts in `results/`.
- Add broker fees and contract multipliers for futures and options.

## Notes

- This is a research scaffold, not production trading infrastructure.
- The notebooks are designed to tell a coherent project story for GitHub, applications, and interviews.
- Generated charts and outputs should be stored in `results/`.
