from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class OptionsStrategyDecision:
    """Suggested options structure for a regime and signal combination."""

    regime: str
    signal: int
    strategy_name: str
    stance: str
    target_dte: int
    max_loss_profile: str
    rationale: str


def select_options_strategy(
    regime_label: str,
    signal: int,
    implied_vol_rank: float | None = None,
) -> OptionsStrategyDecision:
    """Map regime state and directional bias into a baseline options strategy."""

    if pd.isna(regime_label) or pd.isna(signal):
        return OptionsStrategyDecision(
            regime="unclassified",
            signal=0,
            strategy_name="No Trade",
            stance="Flat",
            target_dte=0,
            max_loss_profile="None",
            rationale="There is not enough information yet to classify the regime and assign an options structure.",
        )

    text = regime_label.lower()
    is_bullish = "bullish" in text
    is_bearish = "bearish" in text
    is_high_vol = "high_volatility" in text or (implied_vol_rank is not None and implied_vol_rank >= 0.7)
    is_low_vol = "low_volatility" in text or (implied_vol_rank is not None and implied_vol_rank <= 0.3)

    if signal > 0 and is_bullish:
        if is_high_vol:
            return OptionsStrategyDecision(
                regime=regime_label,
                signal=signal,
                strategy_name="Bull Call Spread",
                stance="Bullish defined-risk spread",
                target_dte=30,
                max_loss_profile="Defined risk",
                rationale="Bullish conditions with elevated volatility favor a call spread over rich outright calls.",
            )
        return OptionsStrategyDecision(
            regime=regime_label,
            signal=signal,
            strategy_name="Long Call",
            stance="Bullish premium-buying",
            target_dte=21,
            max_loss_profile="Premium paid",
            rationale="Low-volatility bullish conditions are a good fit for outright long calls because optionality is relatively cheaper.",
        )

    if signal < 0 and is_bearish:
        if is_low_vol:
            return OptionsStrategyDecision(
                regime=regime_label,
                signal=signal,
                strategy_name="Long Put",
                stance="Bearish premium-buying",
                target_dte=21,
                max_loss_profile="Premium paid",
                rationale="Low-volatility bearish conditions are a good fit for outright long puts because downside convexity is relatively cheaper.",
            )
        return OptionsStrategyDecision(
            regime=regime_label,
            signal=signal,
            strategy_name="Bear Put Spread",
            stance="Bearish defined-risk spread",
            target_dte=30,
            max_loss_profile="Defined risk",
            rationale="High-volatility bearish conditions favor a put spread so the trade still expresses downside while containing premium outlay.",
        )

    if signal == 0:
        if is_high_vol:
            return OptionsStrategyDecision(
                regime=regime_label,
                signal=signal,
                strategy_name="Iron Condor",
                stance="Neutral premium-selling",
                target_dte=25,
                max_loss_profile="Defined risk",
                rationale="Very high-volatility neutral conditions favor harvesting rich premium with a defined-risk range strategy.",
            )
        if is_low_vol:
            return OptionsStrategyDecision(
                regime=regime_label,
                signal=signal,
                strategy_name="Long Straddle",
                stance="Volatility-buying",
                target_dte=45,
                max_loss_profile="Premium paid",
                rationale="Very low-volatility neutral conditions can be a setup for volatility expansion, which suits a long straddle.",
            )
        return OptionsStrategyDecision(
            regime=regime_label,
            signal=signal,
            strategy_name="Calendar Spread",
            stance="Neutral to mildly directional",
            target_dte=30,
            max_loss_profile="Defined risk",
            rationale="When neither direction nor volatility is extreme, a calendar spread offers a balanced neutral structure.",
        )

    if signal > 0:
        return OptionsStrategyDecision(
            regime=regime_label,
            signal=signal,
            strategy_name="Call Spread",
            stance="Counter-regime bullish",
            target_dte=21,
            max_loss_profile="Defined risk",
            rationale="The model is leaning bullish but the regime is mixed, so a capped-risk bullish spread is more disciplined than an outright long call.",
        )

    return OptionsStrategyDecision(
        regime=regime_label,
        signal=signal,
        strategy_name="Put Spread",
        stance="Counter-regime bearish",
        target_dte=21,
        max_loss_profile="Defined risk",
        rationale="The model is leaning bearish but the regime is mixed, so a capped-risk bearish spread is more disciplined than an outright long put.",
    )


def build_strategy_frame(
    regime_labels: pd.Series,
    signals: pd.Series,
    implied_vol_rank: pd.Series | None = None,
) -> pd.DataFrame:
    """Apply strategy selection row by row and return a tabular result."""

    aligned = pd.concat([regime_labels.rename("regime"), signals.rename("signal")], axis=1)
    if implied_vol_rank is not None:
        aligned = aligned.join(implied_vol_rank.rename("implied_vol_rank"))

    decisions = []
    for row in aligned.itertuples():
        decision = select_options_strategy(
            regime_label=row.regime,
            signal=row.signal if pd.isna(row.signal) else int(row.signal),
            implied_vol_rank=getattr(row, "implied_vol_rank", None),
        )
        decisions.append(asdict(decision))

    return pd.DataFrame(decisions, index=aligned.index)
