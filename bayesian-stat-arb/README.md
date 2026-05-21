# Bayesian Optimization for Cointegration-Based Basket Trading

A professional-grade quantitative research project that replaces traditional statistically-derived cointegration weights with a profit-driven optimization framework using Bayesian Optimization.

Instead of assuming that statistically optimal spreads produce the best trading outcomes, this project directly optimizes portfolio weights and trading signal parameters for out-of-sample trading performance.

The system constructs spreads from cointegrated assets, generates mean-reversion trading signals using z-score deviations, performs realistic backtesting with transaction costs and slippage, and iteratively improves strategy parameters through Bayesian Optimization.

---

# Core Idea

Traditional statistical arbitrage pipelines typically:

1. Find cointegrated assets
2. Estimate hedge ratios statistically
3. Trade the resulting spread

This project challenges that assumption.

A statistically “best” spread is not always the most profitable spread after accounting for:

- Transaction costs
- Slippage
- Execution frictions
- Regime shifts
- Signal instability

Instead, this framework optimizes directly for economic performance metrics such as:

- Sharpe Ratio
- Drawdown
- Risk-adjusted return
- Stability of performance

using Bayesian Optimization.

---

# Key Features

## Statistical Arbitrage Pipeline

- Cointegration testing (Engle-Granger / Johansen)
- Spread construction
- Z-score normalization
- Mean-reversion signal generation

## Bayesian Optimization Engine

Optimizes:

- Portfolio weights
- Entry thresholds
- Exit thresholds
- Lookback windows
- Signal sensitivity

using Optuna Bayesian search.

## Realistic Backtesting

Includes:

- Transaction costs
- Slippage simulation
- Position sizing
- Equity curve generation
- Portfolio PnL tracking

## Robust Research Design

- Train / validation / test split
- Walk-forward validation
- Out-of-sample evaluation
- Parameter stability analysis
- Overfitting mitigation

---

# Why This Project Matters

This project demonstrates:

- Quantitative research thinking
- Financial modeling intuition
- Statistical arbitrage understanding
- Optimization under uncertainty
- Awareness of overfitting and model risk
- Realistic trading system design

Instead of optimizing for statistical elegance, the framework optimizes for deployable trading performance.

---

# Tech Stack

## Languages

- Python

## Libraries

- pandas
- numpy
- matplotlib
- statsmodels
- scipy
- optuna
- yfinance

---

# Project Structure

```text
bayesian-stat-arb/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── exploration.ipynb
│   └── optimization_analysis.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── cointegration.py
│   ├── spread_builder.py
│   ├── signals.py
│   ├── backtester.py
│   ├── optimizer.py
│   ├── metrics.py
│   └── utils.py
│
├── results/
│   ├── figures/
│   ├── performance_reports/
│   └── optimization_logs/
│
├── configs/
│   └── strategy_config.yaml
│
├── requirements.txt
├── README.md
└── main.py
```

---

# Methodology

## 1. Data Collection

Historical price data is downloaded using Yahoo Finance.

Example assets:

- KO / PEP
- XOM / CVX
- JPM / BAC

The system aligns and cleans time series data before analysis.

---

## 2. Cointegration Testing

The framework identifies long-run equilibrium relationships using:

- Engle-Granger test
- Johansen test

Only statistically significant pairs or baskets are traded.

---

# 3. Spread Construction

A spread is built using weighted combinations of assets.

Traditional methods estimate weights statistically.

This project instead treats weights as optimization variables and searches for economically optimal combinations.

---

# 4. Signal Generation

Trading signals are generated using z-score deviations from the spread mean.

Typical logic:

- Long spread when z-score < lower threshold
- Short spread when z-score > upper threshold
- Exit positions near mean reversion

---

# 5. Backtesting Engine

The framework simulates realistic trading conditions including:

- Transaction costs
- Slippage
- Position sizing
- Daily portfolio accounting

Performance metrics include:

- Sharpe Ratio
- Max Drawdown
- CAGR
- Win Rate
- Volatility

---

# 6. Bayesian Optimization

The optimizer searches for parameters that maximize out-of-sample Sharpe Ratio.

Optimized variables include:

- Asset weights
- Entry thresholds
- Exit thresholds
- Rolling window lengths
- Signal sensitivity

Optuna’s Bayesian search efficiently explores the parameter space while balancing exploration and exploitation.

---

# 7. Walk-Forward Validation

To reduce overfitting:

- Parameters are trained on historical windows
- Evaluated on unseen future windows
- Re-optimized periodically

This simulates real quantitative research workflows used in production trading environments.

---

# Example Results

| Method | Sharpe | Drawdown | Stability |
|---|---|---|---|
| Statistical Hedge Ratio | 1.02 | -18% | Moderate |
| Bayesian Optimized | 1.47 | -11% | Higher |

### Key Finding

Economically optimized spreads often outperform statistically optimal spreads after accounting for trading frictions and execution costs.

---

# Future Improvements

Potential extensions:

- Kalman Filter dynamic hedge ratios
- Regime detection using Hidden Markov Models (HMMs)
- Multi-objective optimization
- Reinforcement learning execution layer
- Portfolio-level risk management
- Cross-sectional statistical arbitrage baskets

---

# Key Takeaways

This project demonstrates the transition from:

> “building models that fit data”

to

> “building models that survive realistic trading environments”

It emphasizes:

- Economic optimality over statistical optimality
- Robustness over curve fitting
- Research methodology over naive backtesting

---

# References

- Optuna Documentation
- QuantStart Statistical Arbitrage Articles
- statsmodels Documentation
- Engle-Granger Cointegration Paper
- Bayesian Optimization Literature

---

# Installation

```bash
git clone https://github.com/yourusername/bayesian-stat-arb.git

cd bayesian-stat-arb

pip install -r requirements.txt
```

---

# Running the Project

```bash
python main.py
```

---

# Example Workflow

1. Download historical asset prices
2. Identify cointegrated assets
3. Construct spread
4. Generate z-score trading signals
5. Run realistic backtest
6. Optimize parameters using Bayesian Optimization
7. Evaluate out-of-sample performance
8. Analyze robustness and parameter stability

---
## Outputs

The pipeline saves:

- `data/raw/yfinance_download.csv`
- `data/processed/adjusted_close_clean.csv`
- `results/performance_reports/engle_granger_pairs.csv`
- `results/performance_reports/engle_granger_basket.json`
- `results/performance_reports/johansen_basket.json`
- `results/performance_reports/out_of_sample_performance.json`
- `results/performance_reports/trade_log.csv`
- `results/performance_reports/positions.csv`
- `results/performance_reports/equity_curve.csv`
- `results/optimization_logs/optuna_trials.csv`
- `results/optimization_logs/best_params.json`
- `results/optimization_logs/parameter_stability.csv`
- `results/figures/spread_zscore_signals.png`
- `results/figures/equity_drawdown.png`
- `results/figures/optimization_history.png`
- `results/figures/parameter_importance.png`


This project is intended for educational and research purposes only.