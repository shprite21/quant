# Project A — Regime-Aware Multi-Strategy Backtesting System

A professional-grade quantitative research platform that combines multiple trading strategies with Hidden Markov Model (HMM) based market regime detection.

---

## Overview

Financial markets behave differently under different conditions. A trend-following strategy may perform well during strong directional moves, while mean reversion or statistical arbitrage strategies may outperform during range-bound or dislocated markets.

This project builds a regime-aware trading system that:

1. Detects hidden market regimes using a Hidden Markov Model (HMM)
2. Runs multiple strategies simultaneously
3. Dynamically adjusts strategy allocations based on the current regime
4. Evaluates performance using institutional-grade risk metrics

The project is designed to showcase advanced quantitative finance, machine learning, and software engineering skills for MFE applications and quantitative research roles.

---

## Key Features

### Hidden Markov Model (HMM) Regime Detection

* Uses Gaussian HMM to infer latent market states
* Identifies regimes such as:

  * Bull / Trending
  * Bear / Downtrend
  * High Volatility
  * Low Volatility
* Trained on:

  * Daily returns
  * Rolling volatility
  * Momentum indicators

### Trading Strategies

* **Momentum Strategy** — Moving average crossover
* **Mean Reversion Strategy** — Bollinger Band z-score signals
* **Pairs Trading Strategy** — Cointegration-based statistical arbitrage

### Dynamic Allocation Engine

Allocates capital based on the detected regime.

Example:

* Trending market → overweight Momentum
* Sideways market → overweight Mean Reversion
* Dislocated market → overweight Pairs Trading

### Backtesting Engine

* Vectorized backtesting
* Position sizing
* Portfolio aggregation
* Transaction cost modeling
* Benchmark comparison

### Performance Analytics

* CAGR
* Sharpe Ratio
* Sortino Ratio
* Maximum Drawdown
* Calmar Ratio
* Volatility
* Win Rate

### Visualization

* Equity curve
* Drawdown plot
* Regime overlay on prices
* Rolling Sharpe ratio
* Strategy weight evolution

---

## Repository Structure

```text
regime-aware-multi-strategy-backtesting/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── experiments/
│   ├── config.yaml
│   └── run_experiment.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_strategy_research.ipynb
│   ├── 03_regime_detection.ipynb
│   └── 04_results_analysis.ipynb
│
├── outputs/
│   ├── figures/
│   └── reports/
│
├── src/
│   ├── analytics/
│   │   ├── metrics.py
│   │   └── plotting.py
│   │
│   ├── backtester/
│   │   ├── engine.py
│   │   └── portfolio.py
│   │
│   ├── data/
│   │   ├── loader.py
│   │   └── preprocess.py
│   │
│   ├── regime/
│   │   └── hmm_regime_detector.py
│   │
│   ├── strategies/
│   │   ├── base.py
│   │   ├── momentum.py
│   │   ├── mean_reversion.py
│   │   └── pairs_trading.py
│   │
│   └── utils/
│       └── helpers.py
│
├── tests/
│   ├── test_backtester.py
│   ├── test_metrics.py
│   └── test_strategies.py
│
├── .gitignore
├── README.md
├── requirements.txt
└── setup.py
```

---

## Methodology

### 1. Data Preparation

Historical price data is loaded and transformed into features such as:

* Log returns
* Rolling volatility
* Moving averages
* Z-scores

### 2. Regime Detection

The HMM estimates hidden states using market features and labels each date with a regime.

### 3. Signal Generation

Each strategy independently generates signals:

* `+1` = Long
* `0` = Flat
* `-1` = Short

### 4. Regime-Based Allocation

|          Regime | Momentum | Mean Reversion | Pairs Trading |
| --------------: | -------: | -------------: | ------------: |
|      Bull Trend |     0.70 |           0.20 |          0.10 |
|        Sideways |     0.20 |           0.60 |          0.20 |
| High Volatility |     0.10 |           0.20 |          0.70 |

### 5. Portfolio Construction

Portfolio return:

[
R_t = \sum_{i=1}^{N} w_{i,t} r_{i,t}
]

Where:

* (R_t): Portfolio return at time (t)
* (w_{i,t}): Weight of strategy (i)
* (r_{i,t}): Return of strategy (i)

### 6. Performance Evaluation

The combined portfolio is benchmarked against buy-and-hold using risk-adjusted metrics.

---

## Example Performance Metrics

|        Metric | Regime-Aware Portfolio | Buy & Hold |
| ------------: | ---------------------: | ---------: |
|          CAGR |                  18.4% |      11.2% |
|  Sharpe Ratio |                   1.62 |       0.89 |
| Sortino Ratio |                   2.35 |       1.21 |
|  Max Drawdown |                  -9.8% |     -22.4% |
|  Calmar Ratio |                   1.88 |       0.50 |

> These are illustrative results. Actual performance depends on market data and assumptions.

---

## Installation

```bash
git clone https://github.com/shprite21/quant-trading.git
cd quant-trading/regime-aware-multi-strategy-backtesting

python -m venv .venv
source .venv/bin/activate        # Mac/Linux
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

---

## Usage

Run the full experiment pipeline:

```bash
python experiments/run_experiment.py
```

Open Jupyter notebooks:

```bash
jupyter notebook
```

Run unit tests:

```bash
pytest tests/
```

---

## Technologies Used

### Programming

* Python
* NumPy
* pandas

### Quantitative Finance

* statsmodels
* arch

### Machine Learning

* hmmlearn
* scikit-learn

### Visualization

* matplotlib
* seaborn

### Testing

* pytest

---

## Skills Demonstrated

* Quantitative research
* Time series analysis
* Hidden Markov Models
* Statistical arbitrage
* Portfolio construction
* Backtesting
* Risk management
* Software engineering
* Data visualization

---

## Resume Bullet

> Built a regime-aware multi-strategy quantitative trading platform combining momentum, mean reversion, and pairs trading strategies with Hidden Markov Model market state detection, dynamic capital allocation, and risk analytics including Sharpe ratio and maximum drawdown.

---

## Future Enhancements

* Walk-forward optimization
* Bayesian hyperparameter tuning
* Reinforcement learning allocator
* Live paper trading integration
* Docker containerization
* CI/CD with GitHub Actions

---

## References

* Quantitative Trading — Ernest Chan
* Advances in Financial Machine Learning — Marcos López de Prado
* Machine Learning for Asset Managers — Marcos López de Prado

---



This project is for educational and research purposes only and does not constitute financial advice.

---

**Arnaav Raj**

