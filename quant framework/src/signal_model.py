from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


def create_forward_return_targets(
    frame: pd.DataFrame,
    horizon: int = 5,
    positive_threshold: float = 0.02,
    negative_threshold: float | None = None,
    label_mode: str = "binary",
    price_col: str = "close",
) -> pd.DataFrame:
    """Create forward-return labels for supervised learning."""

    if price_col not in frame.columns:
        raise ValueError(f"'{price_col}' was not found in the input DataFrame.")
    if label_mode not in {"binary", "ternary"}:
        raise ValueError("label_mode must be either 'binary' or 'ternary'.")

    forward_return = frame[price_col].shift(-horizon) / frame[price_col] - 1
    target = pd.Series(pd.NA, index=frame.index, dtype="Int64")

    if label_mode == "binary":
        target.loc[forward_return > positive_threshold] = 1
        target.loc[forward_return <= positive_threshold] = 0
    else:
        if negative_threshold is None:
            negative_threshold = positive_threshold
        target.loc[forward_return > positive_threshold] = 1
        target.loc[forward_return < -negative_threshold] = -1
        target.loc[forward_return.between(-negative_threshold, positive_threshold, inclusive="both")] = 0

    return pd.DataFrame(
        {
            "forward_return": forward_return,
            "target": target,
        },
        index=frame.index,
    )


class SwingSignalModel:
    """Primary supervised model with XGBoost and benchmark alternatives."""

    def __init__(
        self,
        model_name: str = "xgboost",
        n_estimators: int = 400,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        min_samples_leaf: int = 10,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ) -> None:
        self.model_name = model_name
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.model = None
        self.feature_names_: list[str] = []
        self.imputation_values_: pd.Series | None = None
        self.class_labels_: list[int] = []
        self.label_to_index_: dict[int, int] = {}
        self.index_to_label_: dict[int, int] = {}

    def fit(self, feature_frame: pd.DataFrame, target: pd.Series) -> "SwingSignalModel":
        """Fit the classifier on aligned features and labels."""

        dataset = feature_frame.copy()
        dataset["target"] = target
        dataset = dataset.replace([np.inf, -np.inf], np.nan).dropna(subset=["target"])

        if dataset.empty:
            raise ValueError("The modeling dataset is empty after alignment.")

        X = dataset.drop(columns="target")
        y = dataset["target"].astype(int)
        if y.nunique() < 2:
            raise ValueError("The target vector must contain at least two classes.")

        self.feature_names_ = list(X.columns)
        self.imputation_values_ = X.median(numeric_only=True)
        X = X.fillna(self.imputation_values_)

        self.class_labels_ = sorted(int(value) for value in pd.unique(y))
        self.label_to_index_ = {label: idx for idx, label in enumerate(self.class_labels_)}
        self.index_to_label_ = {idx: label for label, idx in self.label_to_index_.items()}
        encoded_y = y.map(self.label_to_index_).astype(int)

        self.model = self._build_model(num_classes=len(self.class_labels_))
        self.model.fit(X, encoded_y)
        return self

    def predict(self, feature_frame: pd.DataFrame) -> pd.Series:
        """Predict target classes using the configured classifier."""

        X = self._prepare_features(feature_frame)
        predictions = self.model.predict(X)
        decoded = pd.Series(predictions, index=X.index).map(self.index_to_label_).astype(int)
        return decoded.rename("prediction")

    def predict_proba(self, feature_frame: pd.DataFrame) -> pd.DataFrame:
        """Return class probabilities for the signal model."""

        X = self._prepare_features(feature_frame)
        probabilities = self.model.predict_proba(X)
        probability_columns = [f"class_{label}" for label in self.class_labels_]
        return pd.DataFrame(probabilities, index=X.index, columns=probability_columns)

    def predict_signal(
        self,
        feature_frame: pd.DataFrame,
        upper_threshold: float = 0.55,
        lower_threshold: float = 0.45,
    ) -> pd.Series:
        """Convert predicted probabilities into long, flat, or short signals."""

        probabilities = self.predict_proba(feature_frame)
        return probabilities_to_signal(probabilities, upper_threshold=upper_threshold, lower_threshold=lower_threshold)

    def feature_importance(self) -> pd.Series:
        """Return a sorted feature-importance Series."""

        if not self.feature_names_ or self.model is None:
            raise ValueError("Fit the SwingSignalModel before requesting feature importances.")

        if hasattr(self.model, "feature_importances_"):
            importances = pd.Series(self.model.feature_importances_, index=self.feature_names_)
        elif hasattr(self.model, "coef_"):
            coefficients = np.atleast_2d(self.model.coef_)
            importances = pd.Series(np.abs(coefficients).mean(axis=0), index=self.feature_names_)
        else:
            raise ValueError(f"Feature importance is not available for model '{self.model_name}'.")

        return importances.sort_values(ascending=False)

    def _prepare_features(self, feature_frame: pd.DataFrame) -> pd.DataFrame:
        if not self.feature_names_ or self.imputation_values_ is None or self.model is None:
            raise ValueError("Fit the SwingSignalModel before calling predict.")

        X = feature_frame[self.feature_names_].replace([np.inf, -np.inf], np.nan)
        X = X.fillna(self.imputation_values_)
        return X

    def _build_model(self, num_classes: int):
        if self.model_name == "xgboost":
            if XGBClassifier is None:
                raise ImportError("xgboost is required for model_name='xgboost'. Install it with `pip install xgboost`.")

            model_kwargs = {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "subsample": self.subsample,
                "colsample_bytree": self.colsample_bytree,
                "random_state": self.random_state,
                "n_jobs": -1,
                "eval_metric": "logloss",
            }
            if num_classes > 2:
                model_kwargs["objective"] = "multi:softprob"
                model_kwargs["num_class"] = num_classes
            else:
                model_kwargs["objective"] = "binary:logistic"
            return XGBClassifier(**model_kwargs)

        if self.model_name == "random_forest":
            return RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                class_weight="balanced_subsample",
                random_state=self.random_state,
                n_jobs=-1,
            )

        if self.model_name == "logistic_regression":
            return LogisticRegression(
                class_weight="balanced",
                max_iter=2_000,
                multi_class="auto",
                random_state=self.random_state,
            )

        supported = ", ".join(["xgboost", "random_forest", "logistic_regression"])
        raise ValueError(f"Unsupported model_name '{self.model_name}'. Use one of: {supported}.")


def predictions_to_positions(predictions: pd.Series, allow_short: bool = True) -> pd.Series:
    """Convert class labels into directional exposures."""

    if allow_short:
        return predictions.clip(-1, 1)
    return predictions.clip(lower=0, upper=1)


def probabilities_to_signal(
    probabilities: pd.DataFrame | pd.Series,
    upper_threshold: float = 0.55,
    lower_threshold: float = 0.45,
) -> pd.Series:
    """Convert class probabilities into long, flat, or short trading signals."""

    if isinstance(probabilities, pd.DataFrame):
        multiclass_columns = {"class_-1", "class_0", "class_1"}
        if multiclass_columns.issubset(probabilities.columns):
            column_order = [f"class_{label}" for label in (-1, 0, 1)]
            signal = probabilities[column_order].idxmax(axis=1)
            return signal.str.replace("class_", "", regex=False).astype(int).rename("signal")

        if "class_1" in probabilities.columns:
            positive_probability = probabilities["class_1"]
        else:
            raise ValueError("Expected a 'class_1' column when converting probabilities to directional signals.")
    else:
        positive_probability = pd.Series(probabilities, copy=False)

    signal = pd.Series(0, index=positive_probability.index, dtype=int)
    signal.loc[positive_probability >= upper_threshold] = 1
    signal.loc[positive_probability <= lower_threshold] = -1
    return signal.rename("signal")


def walk_forward_train_predict(
    feature_frame: pd.DataFrame,
    target: pd.Series,
    model_name: str = "xgboost",
    min_train_size: int = 252 * 2,
    test_window: int = 21,
    step_size: int | None = None,
    retrain_window: int | None = None,
    upper_threshold: float = 0.55,
    lower_threshold: float = 0.45,
    **model_kwargs,
) -> pd.DataFrame:
    """Generate out-of-sample predictions using walk-forward retraining."""

    step_size = step_size or test_window
    dataset = feature_frame.copy()
    dataset["target"] = target
    dataset = dataset.replace([np.inf, -np.inf], np.nan).dropna(subset=["target"])

    predictions: list[pd.DataFrame] = []
    for train_end in range(min_train_size, len(dataset), step_size):
        test_end = min(train_end + test_window, len(dataset))
        if test_end <= train_end:
            break

        if retrain_window is None:
            train_slice = dataset.iloc[:train_end]
        else:
            train_slice = dataset.iloc[max(0, train_end - retrain_window):train_end]

        test_slice = dataset.iloc[train_end:test_end]
        if train_slice["target"].nunique() < 2 or test_slice.empty:
            continue

        model = SwingSignalModel(model_name=model_name, **model_kwargs)
        model.fit(train_slice.drop(columns="target"), train_slice["target"])

        probability_frame = model.predict_proba(test_slice.drop(columns="target"))
        output = pd.DataFrame(index=test_slice.index)
        output["target"] = test_slice["target"].astype("Int64")
        output["prediction"] = model.predict(test_slice.drop(columns="target"))
        output = output.join(probability_frame)
        output["signal"] = probabilities_to_signal(
            probability_frame,
            upper_threshold=upper_threshold,
            lower_threshold=lower_threshold,
        )
        if "class_1" in probability_frame.columns:
            output["positive_probability"] = probability_frame["class_1"]
        predictions.append(output)

    if not predictions:
        return pd.DataFrame(columns=["target", "prediction", "signal"])

    return pd.concat(predictions).sort_index()


def evaluate_classification_predictions(
    target: pd.Series,
    predictions: pd.Series,
    probabilities: pd.DataFrame | None = None,
) -> pd.Series:
    """Compute core classification metrics for the signal model."""

    evaluation = pd.concat([target.rename("target"), predictions.rename("prediction")], axis=1).dropna()
    if evaluation.empty:
        raise ValueError("No overlapping non-null target and prediction rows were found.")

    y_true = evaluation["target"].astype(int)
    y_pred = evaluation["prediction"].astype(int)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

    if probabilities is not None:
        probability_frame = probabilities.reindex(evaluation.index)
        unique_labels = sorted(y_true.unique())
        probability_columns = [f"class_{label}" for label in unique_labels]
        if len(unique_labels) == 2 and "class_1" in probability_frame.columns:
            metrics["roc_auc"] = roc_auc_score(y_true, probability_frame["class_1"])
        elif all(column in probability_frame.columns for column in probability_columns):
            metrics["roc_auc"] = roc_auc_score(
                y_true,
                probability_frame[probability_columns],
                multi_class="ovr",
                labels=unique_labels,
            )

    return pd.Series(metrics)
