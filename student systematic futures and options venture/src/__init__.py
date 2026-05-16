"""Core package exports for the trading research framework."""

from .backtester import BacktestResult, WalkForwardBacktester
from .data_loader import MarketDataLoader, align_data_frames
from .features import FeatureEngineer, compute_adx, compute_atr, compute_bollinger_bandwidth, compute_macd, compute_rsi
from .options_strategy import OptionsStrategySelector, StrategyDecision
from .performance import PerformanceAnalyzer
from .regime_detection import RegimeDetectionResult, RegimeDetector
from .risk_management import RiskManager, RiskPlan
from .signal_model import DirectionalSignalModel, PredictionResult, create_targets

__all__ = [
    "BacktestResult",
    "DirectionalSignalModel",
    "FeatureEngineer",
    "MarketDataLoader",
    "OptionsStrategySelector",
    "PerformanceAnalyzer",
    "PredictionResult",
    "RegimeDetectionResult",
    "RegimeDetector",
    "RiskManager",
    "RiskPlan",
    "StrategyDecision",
    "WalkForwardBacktester",
    "align_data_frames",
    "compute_adx",
    "compute_atr",
    "compute_bollinger_bandwidth",
    "compute_macd",
    "compute_rsi",
    "create_targets",
]
