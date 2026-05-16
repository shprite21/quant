# Report Outline

Use this outline to turn the notebook outputs into a polished `report.pdf`.

## 1. Problem Statement

- Why regime awareness matters for swing trading and derivatives selection.
- What decisions the system is designed to support.

## 2. Data

- Universe of assets.
- Sampling frequency and date range.
- Data-cleaning and corporate-action assumptions.

## 3. Feature Engineering

- Momentum, volatility, trend, and mean-reversion features.
- Optional derivatives-specific features such as IV rank and skew.

## 4. Regime Detection

- Hidden Markov Model choice and intuition.
- Regime definitions and summary statistics.
- Transition matrix and persistence analysis.

## 5. Signal Generation

- Labeling logic.
- XGBoost model architecture.
- Benchmark comparison against Logistic Regression and Random Forest.
- Feature-importance interpretation.

## 6. Options Strategy Mapping

- Mapping from regime plus signal to strategy family.
- Rationale behind debit versus credit structures.

## 7. Backtesting and Risk

- Transaction-cost assumptions.
- Benchmark comparison.
- CAGR, Sharpe, Sortino, max drawdown, Calmar, hit rate, and profit factor.

## 8. Limitations and Next Steps

- Look-ahead and survivorship bias controls.
- Walk-forward validation.
- Live-trading infrastructure requirements.
