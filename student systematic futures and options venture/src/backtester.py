"""Walk-forward backtesting and research diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from .options_strategy import OptionsStrategySelector
from .performance import PerformanceAnalyzer
from .signal_model import DirectionalSignalModel
from .utils import walk_forward_windows


@dataclass(slots=True)
class BacktestResult:
    """Container for backtest outputs and diagnostics."""

    predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    performance_summary: pd.DataFrame
    feature_importance: pd.DataFrame


class WalkForwardBacktester:
    """Run walk-forward training, prediction, and strategy simulation."""

    def __init__(
        self,
        model_name: str = "xgboost",
        task_type: str = "binary",
        train_window: int = 252,
        test_window: int = 21,
        step_size: int | None = None,
        probability_threshold: float = 0.55,
        short_probability_threshold: float = 0.45,
        regression_threshold: float = 0.0,
        allow_short: bool = True,
        transaction_cost_bps: float = 2.5,
        slippage_bps: float = 1.0,
        periods_per_year: int = 252,
        risk_free_rate: float = 0.0,
        model_params: dict | None = None,
        strategy_selector: OptionsStrategySelector | None = None,
    ) -> None:
        self.model_name = model_name
        self.task_type = task_type
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        self.probability_threshold = probability_threshold
        self.short_probability_threshold = short_probability_threshold
        self.regression_threshold = regression_threshold
        self.allow_short = allow_short
        self.transaction_cost_bps = transaction_cost_bps
        self.slippage_bps = slippage_bps
        self.periods_per_year = periods_per_year
        self.risk_free_rate = risk_free_rate
        self.model_params = model_params or {}
        self.strategy_selector = strategy_selector

        self.performance_analyzer = PerformanceAnalyzer(
            periods_per_year=self.periods_per_year,
            risk_free_rate=self.risk_free_rate,
        )
        self.last_model_: DirectionalSignalModel | None = None

    def run(
        self,
        data: pd.DataFrame,
        feature_columns: list[str],
        target_column: str,
        price_column: str = "close",
        returns_column: str | None = "daily_return",
    ) -> BacktestResult:
        """Run walk-forward model training and return backtest diagnostics."""

        if price_column not in data.columns:
            raise ValueError(f"price_column '{price_column}' was not found in the input data.")

        prediction_chunks: list[pd.DataFrame] = []
        fold_metrics: list[dict[str, float | int]] = []
        feature_importance_frames: list[pd.DataFrame] = []

        for fold_number, (train_slice, test_slice) in enumerate(
            walk_forward_windows(
                n_samples=len(data),
                train_window=self.train_window,
                test_window=self.test_window,
                step_size=self.step_size,
            ),
            start=1,
        ):
            train_data = data.iloc[train_slice].copy()
            test_data = data.iloc[test_slice].copy()

            model = DirectionalSignalModel(
                model_name=self.model_name,
                task_type=self.task_type,
                model_params=self.model_params,
            )
            model.fit(train_data, feature_columns=feature_columns, target_column=target_column)
            prediction_frame = model.make_prediction_frame(test_data)
            prediction_frame["fold"] = fold_number
            prediction_chunks.append(prediction_frame)

            metrics = model.evaluate(test_data[target_column], prediction_frame)
            metrics["fold"] = fold_number
            fold_metrics.append(metrics)

            fold_importance = model.get_feature_importance()
            fold_importance["fold"] = fold_number
            feature_importance_frames.append(fold_importance)

            self.last_model_ = model

        if not prediction_chunks:
            raise ValueError("No walk-forward folds were generated. Increase the data length or reduce window sizes.")

        predictions = pd.concat(prediction_chunks, axis=0).sort_index()
        predictions = predictions[~predictions.index.duplicated(keep="last")]
        merged = data.join(predictions, how="left")
        merged["position"] = self._positions_from_predictions(merged)

        if returns_column is not None and returns_column in merged.columns:
            asset_returns = merged[returns_column].fillna(0.0)
        else:
            asset_returns = merged[price_column].pct_change().fillna(0.0)
        merged["asset_return"] = asset_returns

        gross_returns = merged["position"].shift(1).fillna(0.0) * merged["asset_return"]
        turnover = merged["position"].diff().abs().fillna(merged["position"].abs())
        trading_cost = turnover * (self.transaction_cost_bps + self.slippage_bps) / 10_000.0

        merged["gross_strategy_return"] = gross_returns
        merged["turnover"] = turnover
        merged["transaction_cost"] = trading_cost
        merged["strategy_return"] = gross_returns - trading_cost

        merged = self._attach_benchmarks(merged, price_column=price_column)

        if self.strategy_selector is not None:
            merged = self.strategy_selector.generate_strategy_signals(merged)

        performance_summary = self._build_performance_summary(merged)
        feature_importance = (
            pd.concat(feature_importance_frames, axis=0)
            .groupby("feature", as_index=False)["importance"]
            .mean()
            .sort_values("importance", ascending=False, ignore_index=True)
        )

        return BacktestResult(
            predictions=merged,
            fold_metrics=pd.DataFrame(fold_metrics),
            performance_summary=performance_summary,
            feature_importance=feature_importance,
        )

    def plot_feature_importance(
        self,
        feature_importance: pd.DataFrame,
        top_n: int = 20,
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        """Plot top feature importances."""

        top_features = feature_importance.head(top_n).sort_values("importance")
        figure, axis = plt.subplots(figsize=(10, 7))
        axis.barh(top_features["feature"], top_features["importance"], color="steelblue")
        axis.set_title(f"Top {top_n} Feature Importances")
        axis.set_xlabel("Importance")
        axis.grid(True, axis="x", alpha=0.3)
        if save_path is not None:
            figure.savefig(save_path, dpi=150, bbox_inches="tight")
        return figure

    def plot_confusion_matrix(
        self,
        result: BacktestResult,
        target_column: str,
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        """Plot a confusion matrix for classification tasks."""

        if self.task_type == "regression":
            raise ValueError("Confusion matrices are only available for classification tasks.")
        frame = result.predictions.dropna(subset=[target_column, "prediction"]).copy()
        figure, axis = plt.subplots(figsize=(6, 6))
        ConfusionMatrixDisplay.from_predictions(
            frame[target_column].astype(int),
            frame["prediction"].astype(int),
            ax=axis,
            colorbar=False,
        )
        axis.set_title("Confusion Matrix")
        if save_path is not None:
            figure.savefig(save_path, dpi=150, bbox_inches="tight")
        return figure

    def plot_roc_curve(
        self,
        result: BacktestResult,
        target_column: str,
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        """Plot ROC curve for binary classification."""

        if self.task_type != "binary":
            raise ValueError("ROC curves are only available for binary classification tasks.")
        frame = result.predictions.dropna(subset=[target_column, "up_probability"]).copy()
        figure, axis = plt.subplots(figsize=(6, 6))
        RocCurveDisplay.from_predictions(
            frame[target_column].astype(int),
            frame["up_probability"],
            ax=axis,
        )
        axis.set_title("ROC Curve")
        if save_path is not None:
            figure.savefig(save_path, dpi=150, bbox_inches="tight")
        return figure

    def plot_regime_overlay(
        self,
        data: pd.DataFrame,
        price_column: str = "close",
        regime_column: str = "regime_id",
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        """Plot price colored by detected regime."""

        if price_column not in data.columns or regime_column not in data.columns:
            raise ValueError("Both price and regime columns are required for regime overlays.")
        figure, axis = plt.subplots(figsize=(12, 6))
        valid = data.dropna(subset=[price_column, regime_column]).copy()
        scatter = axis.scatter(
            valid.index,
            valid[price_column],
            c=valid[regime_column],
            cmap="tab10",
            s=18,
        )
        axis.plot(valid.index, valid[price_column], color="lightgray", linewidth=1.0, alpha=0.8)
        axis.set_title("Price with Regime Overlay")
        axis.set_ylabel(price_column)
        axis.grid(True, alpha=0.3)
        figure.colorbar(scatter, ax=axis, label="Regime")
        if save_path is not None:
            figure.savefig(save_path, dpi=150, bbox_inches="tight")
        return figure

    def plot_transition_matrix_heatmap(
        self,
        transition_matrix: pd.DataFrame,
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        """Plot a transition matrix heatmap."""

        figure, axis = plt.subplots(figsize=(6, 5))
        image = axis.imshow(transition_matrix.values, cmap="viridis")
        axis.set_xticks(range(len(transition_matrix.columns)), transition_matrix.columns, rotation=45)
        axis.set_yticks(range(len(transition_matrix.index)), transition_matrix.index)
        axis.set_title("Transition Matrix Heatmap")
        for row_index in range(transition_matrix.shape[0]):
            for col_index in range(transition_matrix.shape[1]):
                axis.text(
                    col_index,
                    row_index,
                    f"{transition_matrix.iloc[row_index, col_index]:.2f}",
                    ha="center",
                    va="center",
                    color="white",
                )
        figure.colorbar(image, ax=axis)
        if save_path is not None:
            figure.savefig(save_path, dpi=150, bbox_inches="tight")
        return figure

    def plot_shap_summary(
        self,
        data: pd.DataFrame,
        max_samples: int = 500,
        save_path: str | Path | None = None,
    ) -> None:
        """Generate a SHAP summary plot from the most recent fitted model."""

        if self.last_model_ is None:
            raise ValueError("No fitted model is available for SHAP plotting.")
        shap_values, sample = self.last_model_.compute_shap_values(data, max_samples=max_samples)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, sample, show=False)
        if save_path is not None:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

    def _positions_from_predictions(self, data: pd.DataFrame) -> pd.Series:
        positions = pd.Series(0.0, index=data.index)

        if self.task_type == "binary":
            probabilities = data.get("up_probability")
            if probabilities is None:
                return positions
            positions = np.where(probabilities >= self.probability_threshold, 1.0, 0.0)
            if self.allow_short:
                positions = np.where(
                    probabilities <= self.short_probability_threshold,
                    -1.0,
                    positions,
                )
            return pd.Series(positions, index=data.index, name="position")

        if self.task_type == "multiclass":
            confidence = data.get("signal_probability", pd.Series(index=data.index, dtype=float))
            predictions = data.get("prediction", pd.Series(index=data.index, dtype=float))
            long_mask = (predictions == 2) & (confidence >= self.probability_threshold)
            short_mask = (predictions == 0) & (confidence >= self.probability_threshold)
            positions = np.where(long_mask, 1.0, 0.0)
            if self.allow_short:
                positions = np.where(short_mask, -1.0, positions)
            return pd.Series(positions, index=data.index, name="position")

        predictions = data.get("prediction", pd.Series(index=data.index, dtype=float)).fillna(0.0)
        positions = np.where(predictions >= self.regression_threshold, 1.0, 0.0)
        if self.allow_short:
            positions = np.where(predictions <= -self.regression_threshold, -1.0, positions)
        return pd.Series(positions, index=data.index, name="position")

    def _attach_benchmarks(self, data: pd.DataFrame, price_column: str) -> pd.DataFrame:
        frame = data.copy()
        asset_returns = frame["asset_return"].fillna(0.0)

        frame["buy_hold_position"] = 1.0
        frame["buy_hold_return"] = frame["buy_hold_position"].shift(1).fillna(1.0) * asset_returns

        short_sma = frame[price_column].rolling(window=20, min_periods=20).mean()
        long_sma = frame[price_column].rolling(window=50, min_periods=50).mean()
        sma_position = (short_sma > long_sma).astype(float)
        frame["sma_crossover_position"] = sma_position
        frame["sma_crossover_return"] = sma_position.shift(1).fillna(0.0) * asset_returns

        momentum_position = (frame[price_column].pct_change(20) > 0).astype(float)
        frame["momentum_position"] = momentum_position
        frame["momentum_return"] = momentum_position.shift(1).fillna(0.0) * asset_returns
        return frame

    def _build_performance_summary(self, data: pd.DataFrame) -> pd.DataFrame:
        buy_hold = data["buy_hold_return"]
        summary = {
            "strategy": self.performance_analyzer.summarize(
                data["strategy_return"],
                benchmark_returns=buy_hold,
            ),
            "buy_hold": self.performance_analyzer.summarize(buy_hold, benchmark_returns=buy_hold),
            "sma_crossover": self.performance_analyzer.summarize(
                data["sma_crossover_return"],
                benchmark_returns=buy_hold,
            ),
            "momentum": self.performance_analyzer.summarize(
                data["momentum_return"],
                benchmark_returns=buy_hold,
            ),
        }
        return pd.DataFrame(summary).T
