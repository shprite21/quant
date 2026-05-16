# Example Research Summary

This file is an illustrative output artifact that shows how a generated report can look once the notebooks are run on real or paper-trading data.

## Snapshot

- Instrument proxy: `^NSEI`
- Regime model: `Hidden Markov Model`
- Signal model: `XGBoost`
- Prediction horizon: `5 trading days`
- Binary target threshold: `1.5%`

## Example Findings

- The low-volatility trending regime produced the strongest directional hit rate.
- Regime duration was persistent enough to justify using regime state as a supervised feature.
- Strategy returns improved after filtering trades with weak probability and weak trend strength.
- Defined-risk vertical spreads were selected more often than outright long options in high-volatility windows.

## Next Steps

1. Replace the spot proxy with actual futures contract data.
2. Add option-chain snapshots for strike and expiry selection.
3. Extend the backtest from directional proxies to explicit option payoff simulation.
