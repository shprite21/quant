# Monte Carlo Simulation of Stock Returns & Risk

Simulation-based framework to model uncertainty in asset prices and quantify tail risk.

---

## What This Project Demonstrates

* Understanding of probabilistic modeling
* Ability to simulate stochastic processes
* Awareness of model assumptions and limitations

---

## Model Assumption

Daily log returns are modeled as:

$$
r_t \sim \mathcal{N}(\mu, \sigma^2)
$$

Parameters are estimated from historical data.

---

## Methodology

1. Compute daily log returns
2. Estimate mean and volatility
3. Simulate 10,000 future price paths
4. Analyze distribution of returns
5. Compute risk metrics

---

## Risk Metrics

Value at Risk (VaR):

$$
\text{VaR}*\alpha = \text{quantile}*{\alpha}(R)
$$

Expected Shortfall (ES):

$$
\text{ES}*\alpha = \mathbb{E}[R \mid R < \text{VaR}*\alpha]
$$

---

## Key Insight

Financial returns exhibit **fat tails** and volatility clustering.

The normal distribution underestimates extreme losses, leading to overly optimistic risk estimates.

---

## Results

* Generated 10,000 simulated price paths
* Estimated VaR and Expected Shortfall
* Compared empirical vs simulated distributions

---

## Limitations

* Assumes normally distributed returns
* Ignores time-varying volatility
* Does not capture extreme market events

---

## Baseline
Normal distribution
Constant volatility

## Upgraded
Student-t (fat tails)
Empirical validation
Volatility clustering

The upgraded model highlights how normal assumptions underestimate tail risk.