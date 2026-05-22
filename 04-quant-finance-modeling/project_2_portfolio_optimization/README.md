# Mean-Variance Portfolio Optimization

Implementation of Markowitz Portfolio Theory to construct optimal portfolios under the risk–return tradeoff using real market data.

This project builds the efficient frontier, identifies the minimum variance portfolio, and computes the maximum Sharpe ratio (tangency) portfolio, while highlighting practical limitations of mean–variance optimization.

---

## What This Project Demonstrates

- Translation of linear algebra into financial models  
- Understanding of diversification and covariance  
- Awareness of instability in optimization-based portfolios  

---

## Project Visualization

![Efficient Frontier](efficient_frontier.png)

The plot shows:

* Random portfolios colored by Sharpe ratio
* Efficient frontier (optimal portfolios)
* Minimum variance portfolio
* Tangency portfolio
* Capital Market Line

---

## Key Insight

Mean–variance optimization provides a mathematically elegant framework for portfolio construction, but is highly sensitive to input estimates (expected returns and covariance matrix).

Small estimation errors can lead to large changes in optimal weights — a critical limitation in real-world applications.

---

## Mathematical Formulation

Portfolio variance is defined as:

$$
\sigma_p^2 = \mathbf{w}^T \Sigma \mathbf{w}
$$

Subject to:

$$
\sum_i w_i = 1
$$

Where:

* $\mathbf{w}$ = portfolio weights
* $\Sigma$ = covariance matrix of asset returns

The tangency portfolio maximizes the Sharpe ratio:

$$
\text{Sharpe} = \frac{\mathbb{E}[R_p] - R_f}{\sigma_p}
$$

---

## Methodology

### 1. Data Collection

Historical price data is downloaded using `yfinance`.

### 2. Return Computation

Daily returns are computed and annualized assuming 252 trading days.

### 3. Covariance Estimation

The covariance matrix captures interdependencies between assets and drives diversification benefits.

### 4. Optimization

Portfolio weights are obtained by solving a constrained optimization problem using SciPy (SLSQP).

### 5. Efficient Frontier Construction

Optimization is repeated across target returns to trace the efficient frontier.

### 6. Tangency Portfolio

The portfolio with the highest Sharpe ratio is identified and used to construct the Capital Market Line.

---

## Results

* Generated full efficient frontier using real market data
* Identified minimum variance and maximum Sharpe portfolios
* Demonstrated diversification benefits through covariance structure

---

## Backtesting

Train Period: 2022–2023
Test Period: 2024

The optimized portfolio is evaluated against an equal-weight benchmark to assess out-of-sample performance.

---

## Limitations

* Highly sensitive to expected return estimates
* Assumes static covariance structure
* Ignores transaction costs and market impact

These limitations motivate extensions such as regularization and robust optimization.

---

## Technologies Used

Python, NumPy, Pandas, SciPy, Matplotlib, yfinance

---

## Project Structure

```
project_2_portfolio_optimization/
├── mean_variance.ipynb
├── efficient_frontier.png
├── README.md
```

---

## Baseline
Raw covariance
Standard Markowitz

## Upgraded
Covariance shrinkage
Stability improvement

The upgraded model addresses instability in portfolio weights caused by noisy covariance estimates.