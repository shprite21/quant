"""Risk management utilities for systematic trading."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(slots=True)
class RiskPlan:
    """Position-level risk plan derived from account and volatility inputs."""

    stop_loss: float
    unit_risk: float
    position_size: int
    capital_at_risk: float
    notional_exposure: float


@dataclass(slots=True)
class RiskManager:
    """Apply position sizing and stop logic to trades."""

    atr_multiple: float = 2.0
    max_capital_at_risk: float = 0.02
    max_notional_fraction: float = 0.25
    minimum_contracts: int = 0

    def compute_stop_loss(self, entry_price: float, atr: float, side: str = "long") -> float:
        """Compute an ATR-based stop level."""

        if atr < 0:
            raise ValueError("ATR must be non-negative.")
        side_lower = side.lower()
        if side_lower == "long":
            return entry_price - self.atr_multiple * atr
        if side_lower == "short":
            return entry_price + self.atr_multiple * atr
        raise ValueError("side must be either 'long' or 'short'.")

    def compute_position_size(
        self,
        capital: float,
        entry_price: float,
        stop_loss: float,
        contract_multiplier: float = 1.0,
    ) -> int:
        """Determine the maximum position size allowed by risk and notional caps."""

        if capital <= 0:
            return 0
        per_contract_risk = abs(entry_price - stop_loss) * contract_multiplier
        if per_contract_risk <= 0:
            return 0

        risk_budget = capital * self.max_capital_at_risk
        notional_budget = capital * self.max_notional_fraction

        size_from_risk = math.floor(risk_budget / per_contract_risk)
        size_from_notional = math.floor(notional_budget / (entry_price * contract_multiplier))
        position_size = max(
            self.minimum_contracts,
            min(size_from_risk, size_from_notional),
        )
        return max(position_size, 0)

    def build_risk_plan(
        self,
        capital: float,
        entry_price: float,
        atr: float,
        side: str = "long",
        contract_multiplier: float = 1.0,
    ) -> RiskPlan:
        """Create a complete position risk plan."""

        stop_loss = self.compute_stop_loss(entry_price=entry_price, atr=atr, side=side)
        unit_risk = abs(entry_price - stop_loss) * contract_multiplier
        position_size = self.compute_position_size(
            capital=capital,
            entry_price=entry_price,
            stop_loss=stop_loss,
            contract_multiplier=contract_multiplier,
        )
        capital_at_risk = unit_risk * position_size
        notional_exposure = entry_price * contract_multiplier * position_size
        return RiskPlan(
            stop_loss=stop_loss,
            unit_risk=unit_risk,
            position_size=position_size,
            capital_at_risk=capital_at_risk,
            notional_exposure=notional_exposure,
        )

    def enforce_capital_limit(self, proposed_capital_at_risk: float, capital: float) -> bool:
        """Return whether the proposed trade respects the account risk budget."""

        return proposed_capital_at_risk <= capital * self.max_capital_at_risk
