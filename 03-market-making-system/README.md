# Derivatives Pricing + Market Making System

A production-style quantitative trading systems project that simulates a simplified options exchange. The system combines:

- Geometric Brownian Motion stock simulation
- Black-Scholes option pricing
- Greeks computation
- Inventory-aware market making
- Stochastic customer order flow
- Dynamic delta hedging
- Risk analytics
- Python vs C++ performance benchmarking

The project is designed to resemble a realistic sell-side or proprietary trading system where a dealer continuously quotes options, manages inventory exposure, hedges directional risk, and analyzes trading performance.

---

# Core Objectives

This project demonstrates:

- Derivatives pricing knowledge
- Quantitative finance mathematics
- Trading systems engineering
- Market microstructure understanding
- Real-time risk management
- Low-latency optimization using C++
- Production-style software architecture

It is designed for:

- Quant research internships
- Quant developer roles
- Derivatives trading interviews
- MFE/MSFE applications
- Quantitative systems portfolios

---

# Quantitative Concepts

## Geometric Brownian Motion

Underlying stock prices are simulated using:

\[
dS_t = \mu S_t dt + \sigma S_t dW_t
\]

where:

- \( \mu \) = drift
- \( \sigma \) = volatility
- \( W_t \) = Brownian motion

---

## Black-Scholes Pricing

European call option pricing:

\[
C = S N(d_1) - K e^{-rT} N(d_2)
\]

with:

\[
d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)T}{\sigma \sqrt{T}}
\]

\[
d_2 = d_1 - \sigma \sqrt{T}
\]

The system supports:

- European calls
- European puts
- Dynamic repricing through time

---

## Delta Hedging

Portfolio delta exposure is managed dynamically using stock hedging:

\[
\Delta = \frac{\partial V}{\partial S}
\]

The system continuously:

1. Computes portfolio delta
2. Monitors hedge thresholds
3. Executes hedge trades
4. Reduces first-order directional exposure

The simulation also illustrates:

- discrete hedging error
- transaction-cost tradeoffs
- residual gamma exposure
- hedge turnover dynamics

---

# Project Structure

```text
market-making-system/
├── data/
├── python/
│   ├── gbm_simulator.py
│   ├── black_scholes.py
│   ├── greeks.py
│   ├── market_maker.py
│   ├── order_flow.py
│   ├── delta_hedger.py
│   ├── pnl_analysis.py
│   └── benchmark.py
├── cpp/
│   ├── black_scholes.cpp
│   ├── greeks.cpp
│   ├── benchmark.cpp
│   └── Makefile
├── notebooks/
│   └── analysis.ipynb
├── plots/
├── results/
├── requirements.txt
└── README.md
```

---

# System Components

## 1. GBM Stock Simulation

Simulates realistic stock price dynamics using stochastic processes.

Features:

- configurable volatility
- configurable drift
- multiple timesteps
- Monte Carlo-style path generation

Outputs:

- stock price paths
- returns series
- realized volatility estimates

---

## 2. Black-Scholes Pricing Engine

Implements analytical option pricing in:

- Python
- C++

Supports:

- call pricing
- put pricing
- vectorized evaluations
- benchmark throughput testing

Outputs:

- option value
- intrinsic value
- time value

---

## 3. Greeks Engine

Computes:

- Delta
- Gamma
- Vega
- Theta
- Rho

Used for:

- hedging
- inventory risk management
- exposure monitoring
- PnL attribution

---

## 4. Market Maker Engine

The market maker continuously publishes bid/ask quotes:

\[
\text{Bid} = \text{Mid Price} - \text{Spread}
\]

\[
\text{Ask} = \text{Mid Price} + \text{Spread}
\]

Features:

- inventory-aware quote skewing
- spread capture
- adverse selection simulation
- dynamic quote adjustment

Inventory pressure shifts quotes to discourage excessive one-sided positioning.

---

## 5. Order Flow Simulation

Simulates stochastic customer activity:

- buy orders
- sell orders
- no-trade events

This approximates simplified exchange-style interaction between market participants and the dealer.

---

## 6. Inventory Management

Tracks:

- option inventory
- stock hedge inventory
- net delta exposure
- cumulative risk state

The system continuously updates dealer exposure after each simulated trade.

---

## 7. Dynamic Delta Hedging

The hedge engine dynamically trades underlying stock whenever portfolio delta breaches a threshold.

The simulation evaluates:

- hedge quality
- hedge turnover
- transaction costs
- residual exposure
- hedge error over time

This demonstrates realistic continuous risk management mechanics used by derivatives desks.

---

## 8. PnL & Risk Analytics

The analytics engine computes:

- cumulative PnL
- realized volatility
- Sharpe ratio
- drawdowns
- inventory exposure
- hedge error
- portfolio delta trajectories

Visualizations are generated using matplotlib.

---

# Python vs C++ Benchmarking

The project compares Python and C++ pricing performance across large-scale option evaluations.

The C++ engine is optimized for:

- lower latency
- higher throughput
- computational efficiency

Python remains useful for:

- research workflows
- orchestration
- visualization
- analytics pipelines

---

# Simulation Workflow

```text
1. Generate next GBM stock price
2. Reprice options using Black-Scholes
3. Compute Greeks
4. Publish inventory-aware bid/ask quotes
5. Simulate customer order flow
6. Execute option trades
7. Update inventory exposure
8. Recompute portfolio delta
9. Execute hedge trades if needed
10. Mark portfolio to market
11. Record PnL and risk metrics
```

---

# Outputs

## Results (`results/`)

- `gbm_stock_path.csv`
- `simulation_state.csv`
- `option_trades.csv`
- `hedge_trades.csv`
- `risk_metrics.json`
- `benchmark_results.json`

---

## Plots (`plots/`)

- `market_making_dashboard.png`
- `cumulative_pnl.png`
- `inventory_exposure.png`
- `hedge_error.png`
- `realized_volatility.png`
- `option_trade_prices.png`
- `pricing_benchmark.png`

---

# Technologies

## Python

- numpy
- pandas
- scipy
- matplotlib

## C++

- STL
- cmath
- chrono

---

# Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full market-making simulation:

```bash
python python/market_maker.py
```

Run pricing benchmarks:

```bash
python python/benchmark.py --n 200000
```

---

# C++ Build

Automatic benchmark compilation uses:

```bash
g++ -O3 -std=c++17 cpp/benchmark.cpp cpp/black_scholes.cpp cpp/greeks.cpp -o results/cpp_benchmark
```

Manual build:

```bash
cd cpp
make
make run
```

If no compiler is detected, the Python benchmark still executes while C++ results are marked unavailable.

---

# Future Extensions

Potential upgrades:

- multi-strike option books
- multi-expiry surfaces
- implied volatility calibration
- stochastic volatility models
- jump-diffusion dynamics
- Avellaneda-Stoikov market making
- exchange latency simulation
- partial fills and queue positioning
- NumPy vectorization / Numba acceleration
- reinforcement learning market maker
- CI pipelines and unit testing

---

Arnaav Raj