# Project A - Regime-Aware Multi-Strategy Backtesting System

Author: Arnaav Raj

## Overview

This repository is a modular quantitative research platform that combines three systematic trading strategies with Hidden Markov Model based regime detection. The system downloads market data, engineers macro-style state features, infers hidden market regimes, reallocates capital dynamically across strategies, and produces a full backtest report with visual diagnostics.

The project is designed to be portfolio-quality for:

- Quant research internship applications
- MFE / MSFE applications
- GitHub portfolio showcase
- Technical interview discussion

## Features

- Daily market data ingestion from Yahoo Finance via `yfinance`
- Config-driven experiment pipeline
- Strategy library with momentum, mean reversion, and pairs trading
- Gaussian HMM market regime detection using `hmmlearn`
- Regime-aware dynamic strategy allocation
- Vectorized backtesting with look-ahead protection
- Transaction cost and turnover modeling
- Institutional-style performance metrics
- Matplotlib reporting and Plotly interactive dashboard export
- Research notebooks for exploration and presentation
- Unit tests for metrics, strategies, and backtester behavior

## Mathematical Foundation

### 1. Momentum

The momentum strategy uses a moving-average crossover rule:

\[
\text{signal}_t =
\begin{cases}
1, & \text{if } MA_{\text{short}, t} > MA_{\text{long}, t} \\
-1, & \text{otherwise}
\end{cases}
\]

### 2. Mean Reversion

For each asset, a rolling z-score is computed:

\[
z_t = \frac{P_t - \mu_t}{\sigma_t}
\]

The strategy enters long positions when \( z_t < -\theta \) and short positions when \( z_t > \theta \), then exits when the z-score normalizes.

### 3. Pairs Trading

For a correlated pair \( (X_t, Y_t) \), the rolling hedge ratio is estimated via OLS:

\[
Y_t = \alpha_t + \beta_t X_t + \varepsilon_t
\]

The spread is:

\[
S_t = Y_t - \beta_t X_t
\]

The strategy trades the spread based on its rolling z-score.

### 4. Hidden Markov Model Regime Detection

The latent regime process is modeled with a Gaussian Hidden Markov Model using the following feature vector:

- Daily return
- 20-day rolling volatility
- 50-day minus 200-day moving-average spread
- 20-day momentum

The model estimates:

- Hidden states
- State probabilities
- State transition matrix

States are labeled heuristically as:

- `Bull`
- `Bear`
- `High Volatility`

### 5. Portfolio Construction

Strategy capital weights are updated daily according to the inferred regime. Example mapping:

- `Bull`: Momentum 70%, Mean Reversion 20%, Pairs Trading 10%
- `Bear`: Momentum 20%, Mean Reversion 50%, Pairs Trading 30%
- `High Volatility`: Momentum 10%, Mean Reversion 40%, Pairs Trading 50%

## Repository Structure

```text
project-a/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   ├── analytics/
│   ├── backtester/
│   ├── data/
│   ├── regime/
│   ├── strategies/
│   └── utils/
├── experiments/
├── tests/
├── outputs/
│   ├── figures/
│   └── reports/
├── requirements.txt
├── README.md
├── .gitignore
└── setup.py
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run the full experiment from the repository root:

```bash
python experiments/run_experiment.py
```

Optionally point to a custom config:

```bash
python experiments/run_experiment.py --config experiments/config.yaml
```

## Workflow

1. Download and cache daily price data.
2. Clean prices and compute research features.
3. Generate strategy positions.
4. Fit the Gaussian HMM and infer market regimes.
5. Map inferred regimes to strategy capital weights.
6. Backtest the dynamically allocated portfolio.
7. Compute metrics and save reports.
8. Review figures and notebooks for analysis.

## Example Outputs

Running the experiment produces:

- `outputs/figures/equity_curve.png`
- `outputs/figures/drawdown.png`
- `outputs/figures/rolling_sharpe_63d.png`
- `outputs/figures/regime_classification.png`
- `outputs/figures/strategy_cumulative_returns.png`
- `outputs/figures/transition_matrix_heatmap.png`
- `outputs/reports/performance_metrics.csv`
- `outputs/reports/daily_results.csv`
- `outputs/reports/performance_dashboard.html`

## Configuration

The main experiment settings live in `experiments/config.yaml` and include:

- Tickers and benchmark
- Date range
- Strategy parameters
- HMM parameters
- Transaction cost assumptions
- Output locations

## Research Notebooks

The notebook suite is intended for exploratory analysis and presentation:

- `01_data_exploration.ipynb`
- `02_strategy_research.ipynb`
- `03_regime_detection.ipynb`
- `04_results_analysis.ipynb`

## Skills Demonstrated

- Quantitative research workflow design
- Time-series preprocessing and feature engineering
- Strategy signal modeling
- Market regime detection with probabilistic models
- Portfolio construction and backtesting
- Performance analytics and visualization
- Python software engineering, testing, and reproducibility

## Relevance for MFE Applications

This project highlights the blend of statistical modeling, portfolio construction, and production-style engineering expected in financial engineering programs and quantitative internships. It demonstrates:

- Practical use of latent-state models in asset allocation
- Implementation of systematic strategies with realistic execution assumptions
- Familiarity with financial performance diagnostics
- Ability to turn research ideas into reusable software

## References

- Hamilton, J. D. (1989). A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle.
- Rabiner, L. R. (1989). A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition.
- `hmmlearn` documentation
- `statsmodels` documentation
- Yahoo Finance market data via `yfinance`

## Notes

- The experiment uses one-day signal shifting to avoid look-ahead bias.
- Transaction costs default to 5 basis points per unit of turnover.
- If Yahoo Finance access is unavailable, the loader falls back to cached raw data when present.
