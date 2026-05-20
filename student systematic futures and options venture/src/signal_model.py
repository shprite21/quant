"""Supervised models for directional return prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor


@dataclass(slots=True)
class PredictionResult:
    """Container for model predictions and diagnostics."""

    predictions: pd.DataFrame
    feature_importance: pd.DataFrame
    metrics: dict[str, float]


def create_targets(
    data: pd.DataFrame,
    horizon: int = 5,
    return_threshold: float = 0.015,
    lower_threshold: float | None = None,
    upper_threshold: float | None = None,
    price_column: str = "close",
) -> pd.DataFrame:
    """Create binary, multiclass, and regression targets from forward returns."""

    if price_column not in data.columns:
        raise ValueError(f"price_column '{price_column}' was not found in the input data.")

    frame = data.copy()
    frame["forward_return"] = frame[price_column].shift(-horizon) / frame[price_column] - 1.0
    lower = -return_threshold if lower_threshold is None else lower_threshold
    upper = return_threshold if upper_threshold is None else upper_threshold

    frame["target_binary"] = (frame["forward_return"] > return_threshold).astype(float)
    frame.loc[frame["forward_return"].isna(), "target_binary"] = np.nan

    multiclass = np.select(
        [
            frame["forward_return"] < lower,
            frame["forward_return"] > upper,
        ],
        [
            0.0,  # bearish
            2.0,  # bullish
        ],
        default=1.0,  # neutral
    )
    frame["target_multiclass"] = multiclass
    frame.loc[frame["forward_return"].isna(), "target_multiclass"] = np.nan
    frame["target_regression"] = frame["forward_return"]
    return frame


class DirectionalSignalModel:
    """Train supervised models for directional market forecasting."""

    def __init__(
        self,
        model_name: str = "xgboost",
        task_type: str = "binary",
        random_state: int = 42,
        model_params: dict[str, Any] | None = None,
    ) -> None:
        self.model_name = model_name.lower()
        self.task_type = task_type.lower()
        self.random_state = random_state
        self.model_params = model_params or {}

        self.model: Pipeline | None = None
        self.feature_columns_: list[str] = []
        self.target_column_: str = ""
        self.classes_: np.ndarray | None = None

    def fit(
        self,
        data: pd.DataFrame,
        feature_columns: list[str],
        target_column: str,
    ) -> "DirectionalSignalModel":
        """Fit the configured supervised model."""

        clean_data = self._clean_training_data(data, feature_columns, target_column)
        X = clean_data[feature_columns]
        y = clean_data[target_column]
        if self.task_type in {"binary", "multiclass"}:
            y = y.astype(int)

        self.feature_columns_ = feature_columns.copy()
        self.target_column_ = target_column
        self.model = self._build_model(y)
        self.model.fit(X, y)

        if self.task_type in {"binary", "multiclass"}:
            estimator = self.model.named_steps["model"]
            self.classes_ = getattr(estimator, "classes_", None)
        return self

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Generate point predictions."""

        self._validate_fitted()
        X = data[self.feature_columns_].copy()
        predictions = self.model.predict(X)
        return np.asarray(predictions)

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """Generate class probabilities for classification tasks."""

        self._validate_fitted()
        if self.task_type == "regression":
            raise ValueError("predict_proba is not available for regression models.")
        probabilities = self.model.predict_proba(data[self.feature_columns_].copy())
        return np.asarray(probabilities)

    def fit_predict(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        feature_columns: list[str],
        target_column: str,
    ) -> PredictionResult:
        """Fit on a training set and score predictions on a holdout set."""

        self.fit(train_data, feature_columns=feature_columns, target_column=target_column)
        predictions = self.make_prediction_frame(test_data)
        metrics = self.evaluate(test_data[target_column], predictions)
        feature_importance = self.get_feature_importance()
        return PredictionResult(
            predictions=predictions,
            feature_importance=feature_importance,
            metrics=metrics,
        )

    def make_prediction_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a data frame containing predictions and probabilities."""

        self._validate_fitted()
        frame = pd.DataFrame(index=data.index)
        point_predictions = self.predict(data)
        frame["prediction"] = point_predictions

        if self.task_type in {"binary", "multiclass"}:
            probabilities = self.predict_proba(data)
            for class_index in range(probabilities.shape[1]):
                frame[f"probability_{class_index}"] = probabilities[:, class_index]
            frame["signal_probability"] = probabilities.max(axis=1)
            if self.task_type == "binary":
                positive_class_index = 1 if probabilities.shape[1] > 1 else 0
                frame["up_probability"] = probabilities[:, positive_class_index]
        return frame

    def get_feature_importance(self) -> pd.DataFrame:
        """Extract feature importance or coefficient-based rankings."""

        self._validate_fitted()
        estimator = self.model.named_steps["model"]

        if hasattr(estimator, "feature_importances_"):
            importance = np.asarray(estimator.feature_importances_, dtype=float)
        elif hasattr(estimator, "coef_"):
            coefficients = np.asarray(estimator.coef_, dtype=float)
            importance = np.abs(coefficients).mean(axis=0)
        else:
            importance = np.zeros(len(self.feature_columns_), dtype=float)

        feature_importance = pd.DataFrame(
            {
                "feature": self.feature_columns_,
                "importance": importance,
            }
        ).sort_values("importance", ascending=False, ignore_index=True)
        return feature_importance

    def evaluate(
        self,
        y_true: pd.Series | np.ndarray,
        prediction_frame: pd.DataFrame,
    ) -> dict[str, float]:
        """Compute task-appropriate evaluation metrics."""

        y_true_array = np.asarray(y_true, dtype=float)
        valid_mask = ~np.isnan(y_true_array)
        y_true_array = y_true_array[valid_mask]
        predictions = prediction_frame.loc[valid_mask, "prediction"].to_numpy(dtype=float)

        if self.task_type == "regression":
            rmse = float(np.sqrt(mean_squared_error(y_true_array, predictions)))
            mae = float(mean_absolute_error(y_true_array, predictions))
            r2 = float(r2_score(y_true_array, predictions))
            return {"rmse": rmse, "mae": mae, "r2": r2}

        metrics = {
            "accuracy": float(accuracy_score(y_true_array, predictions)),
            "precision": float(
                precision_score(
                    y_true_array,
                    predictions,
                    average="binary" if self.task_type == "binary" else "macro",
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    y_true_array,
                    predictions,
                    average="binary" if self.task_type == "binary" else "macro",
                    zero_division=0,
                )
            ),
            "f1": float(
                f1_score(
                    y_true_array,
                    predictions,
                    average="binary" if self.task_type == "binary" else "macro",
                    zero_division=0,
                )
            ),
        }
        probability_columns = [column for column in prediction_frame.columns if column.startswith("probability_")]
        if probability_columns:
            probabilities = prediction_frame.loc[valid_mask, probability_columns].to_numpy(dtype=float)
            metrics["log_loss"] = float(log_loss(y_true_array, probabilities))
            if self.task_type == "binary" and probabilities.shape[1] > 1:
                metrics["roc_auc"] = float(roc_auc_score(y_true_array, probabilities[:, 1]))
        return metrics

    def compute_shap_values(
        self,
        data: pd.DataFrame,
        max_samples: int = 1000,
    ) -> tuple[Any, pd.DataFrame]:
        """Compute SHAP values for supported model families."""

        self._validate_fitted()
        sample = data[self.feature_columns_].copy().replace([np.inf, -np.inf], np.nan)
        sample = sample.dropna().head(max_samples)
        if sample.empty:
            raise ValueError("No rows are available for SHAP computation.")

        transformed = self.model.named_steps["imputer"].transform(sample)
        estimator = self.model.named_steps["model"]

        if self.model_name in {"xgboost", "random_forest"}:
            explainer = shap.TreeExplainer(estimator)
            shap_values = explainer.shap_values(transformed)
        elif self.model_name == "logistic_regression":
            scaler = self.model.named_steps["scaler"]
            transformed = scaler.transform(transformed)
            explainer = shap.LinearExplainer(estimator, transformed)
            shap_values = explainer.shap_values(transformed)
        else:
            raise ValueError(f"SHAP is not implemented for model '{self.model_name}'.")
        return shap_values, sample

    def _build_model(self, y: pd.Series) -> Pipeline:
        steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]

        if self.model_name == "xgboost":
            estimator = self._build_xgboost_estimator(y)
        elif self.model_name == "random_forest":
            estimator = self._build_random_forest_estimator(y)
        elif self.model_name == "logistic_regression":
            if self.task_type == "regression":
                raise ValueError("Logistic regression is only valid for classification tasks.")
            steps.append(("scaler", StandardScaler()))
            estimator = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                multi_class="auto",
                random_state=self.random_state,
                **self.model_params,
            )
        else:
            raise ValueError(
                "model_name must be one of {'xgboost', 'random_forest', 'logistic_regression'}."
            )

        steps.append(("model", estimator))
        return Pipeline(steps)

    def _build_xgboost_estimator(self, y: pd.Series) -> XGBClassifier | XGBRegressor:
        common_params = {
            "n_estimators": 250,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": self.random_state,
        }
        common_params.update(self.model_params)

        if self.task_type == "binary":
            return XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                **common_params,
            )
        if self.task_type == "multiclass":
            num_classes = int(pd.Series(y).nunique())
            return XGBClassifier(
                objective="multi:softprob",
                eval_metric="mlogloss",
                num_class=num_classes,
                **common_params,
            )
        return XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            **common_params,
        )

    def _build_random_forest_estimator(
        self,
        y: pd.Series,
    ) -> RandomForestClassifier | RandomForestRegressor:
        params = {"n_estimators": 300, "max_depth": 6, "random_state": self.random_state}
        params.update(self.model_params)
        if self.task_type == "regression":
            return RandomForestRegressor(**params)
        return RandomForestClassifier(class_weight="balanced", **params)

    @staticmethod
    def _clean_training_data(
        data: pd.DataFrame,
        feature_columns: list[str],
        target_column: str,
    ) -> pd.DataFrame:
        required_columns = feature_columns + [target_column]
        subset = data[required_columns].copy()
        subset = subset.replace([np.inf, -np.inf], np.nan).dropna(subset=[target_column])
        if subset.empty:
            raise ValueError("No training samples remain after dropping missing targets.")
        return data.loc[subset.index].copy()

    def _validate_fitted(self) -> None:
        if self.model is None or not self.feature_columns_:
            raise ValueError("The signal model must be fitted before use.")
