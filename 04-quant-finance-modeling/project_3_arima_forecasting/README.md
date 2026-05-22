# ARIMA-Based Price Forecasting

Time-series modeling framework to forecast asset prices and evaluate predictive performance.

---

## What This Project Demonstrates

* Understanding of time-series models
* Ability to evaluate forecasts rigorously
* Awareness of predictive limitations in financial markets

---

## Methodology

1. Check stationarity (ADF test)
2. Apply differencing if required
3. Fit ARIMA model
4. Generate forecasts
5. Evaluate performance

---

## Baseline Comparison

Naive model:

$$
\hat{y}*t = y*{t-1}
$$

Used as a benchmark to evaluate ARIMA performance.

---

## Evaluation Metrics

* RMSE
* MAE

---

## Key Insight

In financial markets, simple models often perform comparably to complex ones.

ARIMA models may fail to significantly outperform naive baselines due to market efficiency.

---

## Results

* Generated forecasts using ARIMA
* Compared against naive baseline
* Evaluated prediction errors

---

## Residual Analysis

Residuals are analyzed to check for:

* Autocorrelation
* Non-random patterns

---

## Limitations

* Assumes linear relationships
* Sensitive to non-stationarity
* Limited predictive power in efficient markets

---

## Baseline
Single train-test split
One-shot forecast

## Upgraded
Rolling forecast
Baseline comparison

The upgraded model addresses instability in portfolio weights caused by noisy covariance estimates.