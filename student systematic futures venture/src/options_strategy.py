"""Rule-based options strategy selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class StrategyDecision:
    """Container for a single strategy recommendation."""

    strategy: str
    directional_bias: str
    conviction: float
    rationale: str


class OptionsStrategySelector:
    """Map regimes and directional forecasts to options strategy templates."""

    def __init__(
        self,
        bullish_probability_threshold: float = 0.60,
        bearish_probability_threshold: float = 0.40,
        high_volatility_threshold: float = 0.025,
        low_volatility_threshold: float = 0.012,
        strong_trend_threshold: float = 0.15,
        weak_trend_threshold: float = 0.03,
        regime_bias_map: dict[int, str] | None = None,
    ) -> None:
        self.bullish_probability_threshold = bullish_probability_threshold
        self.bearish_probability_threshold = bearish_probability_threshold
        self.high_volatility_threshold = high_volatility_threshold
        self.low_volatility_threshold = low_volatility_threshold
        self.strong_trend_threshold = strong_trend_threshold
        self.weak_trend_threshold = weak_trend_threshold
        self.regime_bias_map = regime_bias_map or {}

    def select_strategy(
        self,
        regime_id: int | float | None,
        bullish_probability: float,
        volatility_level: float,
        trend_strength: float,
    ) -> StrategyDecision:
        """Select an options strategy from signal and market-state inputs."""

        regime_bias = self.regime_bias_map.get(int(regime_id), "unknown") if pd.notna(regime_id) else "unknown"
        conviction = abs(bullish_probability - 0.5) * 2.0

        if bullish_probability >= self.bullish_probability_threshold:
            if volatility_level >= self.high_volatility_threshold:
                return StrategyDecision(
                    strategy="Bull Call Spread",
                    directional_bias="bullish",
                    conviction=conviction,
                    rationale=(
                        "Bullish signal with elevated volatility favors a defined-risk "
                        "bullish vertical spread."
                    ),
                )
            if trend_strength >= self.strong_trend_threshold or regime_bias == "bullish_trend":
                return StrategyDecision(
                    strategy="Long Call",
                    directional_bias="bullish",
                    conviction=conviction,
                    rationale="Bullish signal and strong trend support outright upside optionality.",
                )
            return StrategyDecision(
                strategy="Bull Call Spread",
                directional_bias="bullish",
                conviction=conviction,
                rationale="Bullish view with moderate trend is expressed more efficiently via a call spread.",
            )

        if bullish_probability <= self.bearish_probability_threshold:
            if volatility_level >= self.high_volatility_threshold:
                return StrategyDecision(
                    strategy="Bear Put Spread",
                    directional_bias="bearish",
                    conviction=conviction,
                    rationale=(
                        "Bearish signal with elevated volatility favors a defined-risk "
                        "bearish put spread over naked premium."
                    ),
                )
            if trend_strength >= self.strong_trend_threshold or regime_bias == "bearish_trend":
                return StrategyDecision(
                    strategy="Long Put",
                    directional_bias="bearish",
                    conviction=conviction,
                    rationale="Bearish signal and strong downtrend support long downside optionality.",
                )
            return StrategyDecision(
                strategy="Bear Put Spread",
                directional_bias="bearish",
                conviction=conviction,
                rationale="Moderate bearish conditions favor a capital-efficient put spread.",
            )

        if volatility_level >= self.high_volatility_threshold and trend_strength <= self.weak_trend_threshold:
            return StrategyDecision(
                strategy="Long Straddle",
                directional_bias="neutral",
                conviction=conviction,
                rationale=(
                    "Signal uncertainty combined with high volatility and weak trend suggests "
                    "a breakout-style long volatility structure."
                ),
            )

        return StrategyDecision(
            strategy="Iron Condor",
            directional_bias="neutral",
            conviction=conviction,
            rationale=(
                "Low directional conviction and contained trend favor collecting premium "
                "through a market-neutral defined-risk structure."
            ),
        )

    def generate_strategy_signals(
        self,
        data: pd.DataFrame,
        probability_column: str = "up_probability",
        volatility_column: str = "atr_percent",
        trend_column: str = "trend_strength",
        regime_column: str = "regime_id",
    ) -> pd.DataFrame:
        """Apply the rules to each row in a data frame."""

        required_columns = {probability_column, volatility_column, trend_column}
        missing = sorted(required_columns.difference(data.columns))
        if missing:
            raise ValueError(f"Missing columns required for strategy selection: {missing}")

        frame = data.copy()
        strategies: list[str] = []
        biases: list[str] = []
        convictions: list[float] = []
        rationales: list[str] = []

        for _, row in frame.iterrows():
            bullish_probability = self._extract_bullish_probability(row, probability_column=probability_column)
            volatility_level = float(row.get(volatility_column, np.nan))
            trend_strength = float(row.get(trend_column, 0.0))
            regime_id = row.get(regime_column, np.nan)

            if np.isnan(bullish_probability) or np.isnan(volatility_level):
                strategies.append(np.nan)
                biases.append(np.nan)
                convictions.append(np.nan)
                rationales.append(np.nan)
                continue

            decision = self.select_strategy(
                regime_id=regime_id,
                bullish_probability=bullish_probability,
                volatility_level=volatility_level,
                trend_strength=trend_strength,
            )
            strategies.append(decision.strategy)
            biases.append(decision.directional_bias)
            convictions.append(decision.conviction)
            rationales.append(decision.rationale)

        frame["strategy"] = strategies
        frame["strategy_bias"] = biases
        frame["strategy_conviction"] = convictions
        frame["strategy_rationale"] = rationales
        return frame

    @staticmethod
    def _extract_bullish_probability(row: pd.Series, probability_column: str) -> float:
        if probability_column in row and pd.notna(row[probability_column]):
            return float(row[probability_column])
        if "probability_2" in row and pd.notna(row["probability_2"]):
            return float(row["probability_2"])
        if "signal_probability" in row and pd.notna(row["signal_probability"]):
            prediction = float(row.get("prediction", np.nan))
            signal_probability = float(row["signal_probability"])
            if prediction == 1.0:
                return signal_probability
            if prediction == 0.0:
                return 1.0 - signal_probability
        return np.nan
