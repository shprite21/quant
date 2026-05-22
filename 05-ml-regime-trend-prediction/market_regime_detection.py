"""Market Regime Detection and Stock Trend Prediction Using Machine Learning.

This script downloads NIFTY 50 data from Yahoo Finance, engineers technical
features and market regimes, trains three classification models, evaluates
them, and saves the processed dataset, figures, and model comparison table.
"""

from __future__ import annotations

import sys
import warnings
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_PACKAGES = PROJECT_ROOT / ".python_packages"

if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
START_DATE = "2015-01-01"
END_DATE = (date.today() + timedelta(days=1)).isoformat()
TICKER = "^NSEI"
REGIME_COLORS = {"bullish": "#2ca02c", "sideways": "#f2b134", "bearish": "#d62728"}

FIGURES_DIR = PROJECT_ROOT / "figures"
YFINANCE_CACHE_DIR = PROJECT_ROOT / ".cache" / "yfinance"
DATASET_PATH = PROJECT_ROOT / "processed_dataset.csv"
COMPARISON_PATH = PROJECT_ROOT / "comparison_results.csv"
PREDICTIONS_PATH = PROJECT_ROOT / "test_predictions.csv"
RF_IMPORTANCE_PATH = PROJECT_ROOT / "random_forest_feature_importance.csv"
XGB_IMPORTANCE_PATH = PROJECT_ROOT / "xgboost_feature_importance.csv"


def ensure_output_directories() -> None:
    """Create folders used by the project outputs."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    YFINANCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute the Relative Strength Index (RSI)."""
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.rolling(window=period, min_periods=period).mean()
    avg_loss = losses.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss != 0, 100)
    rsi = rsi.where(avg_gain != 0, 0)
    return rsi


def download_market_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Download and clean NIFTY 50 historical data from Yahoo Finance."""
    print(f"Downloading {ticker} data from {start_date} to {end_date}...")
    yf.set_tz_cache_location(str(YFINANCE_CACHE_DIR))

    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if data.empty:
        raise RuntimeError("No data was returned from Yahoo Finance.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    data = data.rename(columns=rename_map)
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()

    numeric_columns = data.select_dtypes(include=[np.number]).columns
    data[numeric_columns] = data[numeric_columns].ffill().bfill()
    return data


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create technical indicators, labels, and market regime features."""
    df = data.copy()

    df["daily_return"] = df["close"].pct_change()
    df["sma_20"] = df["close"].rolling(window=20, min_periods=20).mean()
    df["sma_50"] = df["close"].rolling(window=50, min_periods=50).mean()
    df["rsi_14"] = compute_rsi(df["close"], period=14)

    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    rolling_std_20 = df["close"].rolling(window=20, min_periods=20).std()
    upper_band = df["sma_20"] + (2 * rolling_std_20)
    lower_band = df["sma_20"] - (2 * rolling_std_20)
    df["bollinger_band_width"] = (upper_band - lower_band) / df["sma_20"]

    df["rolling_volatility_20"] = df["daily_return"].rolling(window=20, min_periods=20).std()

    df["next_close"] = df["close"].shift(-1)
    df["target"] = np.where(
        df["next_close"].isna(),
        np.nan,
        (df["next_close"] > df["close"]).astype(int),
    )

    volatility_median = df["rolling_volatility_20"].median(skipna=True)
    bullish_mask = (df["close"] > df["sma_50"]) & (
        df["rolling_volatility_20"] < volatility_median
    )
    bearish_mask = (df["close"] < df["sma_50"]) & (
        df["rolling_volatility_20"] > volatility_median
    )

    df["regime"] = np.select(
        [bullish_mask, bearish_mask],
        ["bullish", "bearish"],
        default="sideways",
    )
    df["regime_code"] = df["regime"].map({"bearish": -1, "sideways": 0, "bullish": 1})

    regime_dummies = pd.get_dummies(df["regime"], prefix="regime", dtype=int)
    expected_dummy_columns = ["regime_bearish", "regime_bullish", "regime_sideways"]
    regime_dummies = regime_dummies.reindex(columns=expected_dummy_columns, fill_value=0)
    df = pd.concat([df, regime_dummies], axis=1)

    model_columns = [
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "daily_return",
        "sma_20",
        "sma_50",
        "rsi_14",
        "macd",
        "macd_signal",
        "bollinger_band_width",
        "rolling_volatility_20",
        "regime_bearish",
        "regime_bullish",
        "regime_sideways",
        "target",
    ]

    processed = df.dropna(subset=model_columns).copy()
    processed = processed.drop(columns=["next_close"])
    processed["target"] = processed["target"].astype(int)
    processed.index.name = "date"
    return processed


def save_processed_dataset(processed_data: pd.DataFrame) -> None:
    """Save the fully processed modeling dataset."""
    output = processed_data.reset_index()
    output.to_csv(DATASET_PATH, index=False)


def save_test_predictions(predictions: pd.DataFrame) -> None:
    """Save model predictions on the chronological test split."""
    predictions.reset_index().to_csv(PREDICTIONS_PATH, index=False)


def plot_closing_price(raw_data: pd.DataFrame) -> None:
    plt.figure(figsize=(14, 6))
    plt.plot(raw_data.index, raw_data["close"], color="navy", linewidth=1.6)
    plt.title("NIFTY 50 Closing Price")
    plt.xlabel("Date")
    plt.ylabel("Close")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "closing_price_chart.png", dpi=300)
    plt.close()


def plot_return_histogram(processed_data: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    sns.histplot(processed_data["daily_return"], bins=50, kde=True, color="teal")
    plt.title("Distribution of Daily Returns")
    plt.xlabel("Daily Return")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "return_histogram.png", dpi=300)
    plt.close()


def plot_price_with_regimes(processed_data: pd.DataFrame) -> None:
    plt.figure(figsize=(15, 7))
    plt.plot(processed_data.index, processed_data["close"], color="black", linewidth=1.5, label="Close")
    plt.plot(processed_data.index, processed_data["sma_20"], color="#1f77b4", linewidth=1.2, label="SMA 20")
    plt.plot(processed_data.index, processed_data["sma_50"], color="#9467bd", linewidth=1.2, label="SMA 50")

    for regime, color in REGIME_COLORS.items():
        regime_slice = processed_data[processed_data["regime"] == regime]
        plt.scatter(
            regime_slice.index,
            regime_slice["close"],
            s=12,
            alpha=0.55,
            color=color,
            label=f"{regime.title()} Regime",
        )

    plt.title("NIFTY 50 Price with SMA Overlays and Market Regimes")
    plt.xlabel("Date")
    plt.ylabel("Close")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "price_with_regimes.png", dpi=300)
    plt.close()


def plot_correlation_heatmap(processed_data: pd.DataFrame) -> None:
    numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
    correlation_matrix = processed_data[numeric_columns].corr()

    plt.figure(figsize=(14, 10))
    sns.heatmap(correlation_matrix, cmap="coolwarm", center=0, linewidths=0.4)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=300)
    plt.close()


def plot_regime_distribution(processed_data: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    order = ["bullish", "sideways", "bearish"]
    sns.countplot(data=processed_data, x="regime", order=order, palette="viridis")
    plt.title("Market Regime Distribution")
    plt.xlabel("Regime")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "regime_distribution.png", dpi=300)
    plt.close()


def plot_indicator_panels(processed_data: pd.DataFrame) -> None:
    volatility_median = processed_data["rolling_volatility_20"].median()
    macd_histogram = processed_data["macd"] - processed_data["macd_signal"]
    histogram_colors = np.where(macd_histogram >= 0, "#2ca02c", "#d62728")

    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

    axes[0].plot(processed_data.index, processed_data["rsi_14"], color="#ff7f0e", linewidth=1.3)
    axes[0].axhline(70, linestyle="--", color="#d62728", linewidth=1)
    axes[0].axhline(30, linestyle="--", color="#2ca02c", linewidth=1)
    axes[0].set_title("RSI (14)")
    axes[0].set_ylabel("RSI")

    axes[1].plot(processed_data.index, processed_data["macd"], color="#1f77b4", linewidth=1.2, label="MACD")
    axes[1].plot(
        processed_data.index,
        processed_data["macd_signal"],
        color="#ff7f0e",
        linewidth=1.2,
        label="Signal",
    )
    axes[1].bar(processed_data.index, macd_histogram, color=histogram_colors, alpha=0.35, label="Histogram")
    axes[1].set_title("MACD and Signal Line")
    axes[1].set_ylabel("Value")
    axes[1].legend(loc="upper left")

    axes[2].plot(
        processed_data.index,
        processed_data["rolling_volatility_20"],
        color="#8c564b",
        linewidth=1.3,
        label="20-Day Volatility",
    )
    axes[2].axhline(volatility_median, linestyle="--", color="gray", linewidth=1, label="Median Volatility")
    axes[2].set_title("20-Day Rolling Volatility")
    axes[2].set_xlabel("Date")
    axes[2].set_ylabel("Volatility")
    axes[2].legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "indicator_panels.png", dpi=300)
    plt.close()


def plot_monthly_return_heatmap(raw_data: pd.DataFrame) -> None:
    monthly_close = raw_data["close"].resample("ME").last()
    monthly_returns = monthly_close.pct_change().dropna()
    monthly_frame = monthly_returns.to_frame(name="monthly_return")
    monthly_frame["year"] = monthly_frame.index.year
    monthly_frame["month"] = monthly_frame.index.strftime("%b")

    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    heatmap_data = monthly_frame.pivot(index="year", columns="month", values="monthly_return")
    heatmap_data = heatmap_data.reindex(columns=month_order)

    plt.figure(figsize=(14, 6))
    sns.heatmap(heatmap_data * 100, annot=True, fmt=".1f", cmap="RdYlGn", center=0, linewidths=0.3)
    plt.title("Monthly Return Heatmap (%)")
    plt.xlabel("Month")
    plt.ylabel("Year")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "monthly_return_heatmap.png", dpi=300)
    plt.close()


def run_eda(raw_data: pd.DataFrame, processed_data: pd.DataFrame) -> None:
    """Generate and save all exploratory data analysis charts."""
    plot_closing_price(raw_data)
    plot_return_histogram(processed_data)
    plot_price_with_regimes(processed_data)
    plot_correlation_heatmap(processed_data)
    plot_regime_distribution(processed_data)
    plot_indicator_panels(processed_data)
    plot_monthly_return_heatmap(raw_data)


def get_feature_columns() -> list[str]:
    return [
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "daily_return",
        "sma_20",
        "sma_50",
        "rsi_14",
        "macd",
        "macd_signal",
        "bollinger_band_width",
        "rolling_volatility_20",
        "regime_bearish",
        "regime_bullish",
        "regime_sideways",
    ]


def split_data(processed_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the data chronologically into 80% train and 20% test."""
    feature_columns = get_feature_columns()
    X = processed_data[feature_columns]
    y = processed_data["target"].astype(int)

    split_index = int(len(processed_data) * 0.8)

    X_train = X.iloc[:split_index].copy()
    X_test = X.iloc[split_index:].copy()
    y_train = y.iloc[:split_index].copy()
    y_test = y.iloc[split_index:].copy()
    return X_train, X_test, y_train, y_test


def build_models() -> dict[str, object]:
    """Create the three required classifiers."""
    logistic_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]
    )

    random_forest_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    xgboost_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return {
        "Logistic Regression": logistic_model,
        "Random Forest": random_forest_model,
        "XGBoost": xgboost_model,
    }


def evaluate_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray | object]], pd.DataFrame]:
    """Train, cross-validate, and evaluate each model."""
    models = build_models()
    time_series_cv = TimeSeriesSplit(n_splits=5)

    results: list[dict[str, float | int | str]] = []
    evaluation_cache: dict[str, dict[str, np.ndarray | object]] = {}
    predictions = pd.DataFrame(index=y_test.index)
    predictions["actual"] = y_test.astype(int)

    for model_name, model in models.items():
        print(f"Training and evaluating {model_name}...")
        model_key = model_name.lower().replace(" ", "_")

        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=time_series_cv,
            scoring="roc_auc",
            n_jobs=1,
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_score)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        results.append(
            {
                "model": model_name,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "roc_auc": roc_auc,
                "cv_roc_auc_mean": float(np.mean(cv_scores)),
                "cv_roc_auc_std": float(np.std(cv_scores)),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )

        evaluation_cache[model_name] = {
            "model": model,
            "y_pred": y_pred,
            "y_score": y_score,
            "confusion_matrix": np.array([[tn, fp], [fn, tp]]),
        }
        predictions[f"{model_key}_prediction"] = y_pred.astype(int)
        predictions[f"{model_key}_probability"] = y_score

    comparison = pd.DataFrame(results).sort_values(by="roc_auc", ascending=False)
    best_model = comparison.iloc[0]["model"]
    best_model_key = best_model.lower().replace(" ", "_")
    predictions["best_model"] = best_model
    predictions["best_model_prediction"] = predictions[f"{best_model_key}_prediction"]
    predictions["best_model_probability"] = predictions[f"{best_model_key}_probability"]
    predictions.index.name = "date"
    comparison.to_csv(COMPARISON_PATH, index=False)
    return comparison, evaluation_cache, predictions


def plot_roc_curves(y_test: pd.Series, evaluation_cache: dict[str, dict[str, np.ndarray | object]]) -> None:
    plt.figure(figsize=(10, 7))

    for model_name, values in evaluation_cache.items():
        y_score = values["y_score"]
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = roc_auc_score(y_test, y_score)
        plt.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Baseline")
    plt.title("ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curves.png", dpi=300)
    plt.close()


def plot_confusion_matrices(evaluation_cache: dict[str, dict[str, np.ndarray | object]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for axis, (model_name, values) in zip(axes, evaluation_cache.items()):
        matrix = values["confusion_matrix"]
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=axis,
        )
        axis.set_title(f"{model_name}\nConfusion Matrix")
        axis.set_xlabel("Predicted Label")
        axis.set_ylabel("True Label")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrices.png", dpi=300)
    plt.close()


def get_feature_importance_series(model: object, feature_names: list[str]) -> pd.Series:
    """Return sorted feature importances for tree-based models."""
    return pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)


def plot_feature_importance(
    importances: pd.Series,
    chart_title: str,
    output_name: str,
) -> None:
    top_features = importances.sort_values(ascending=True).tail(15)

    plt.figure(figsize=(10, 7))
    top_features.plot(kind="barh", color="steelblue")
    plt.title(chart_title)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / output_name, dpi=300)
    plt.close()


def save_feature_importance_plots(
    evaluation_cache: dict[str, dict[str, np.ndarray | object]],
    feature_names: list[str],
) -> None:
    random_forest_model = evaluation_cache["Random Forest"]["model"]
    xgboost_model = evaluation_cache["XGBoost"]["model"]
    rf_importances = get_feature_importance_series(random_forest_model, feature_names)
    xgb_importances = get_feature_importance_series(xgboost_model, feature_names)

    rf_importances.rename_axis("feature").reset_index(name="importance").to_csv(RF_IMPORTANCE_PATH, index=False)
    xgb_importances.rename_axis("feature").reset_index(name="importance").to_csv(XGB_IMPORTANCE_PATH, index=False)

    plot_feature_importance(
        importances=rf_importances,
        chart_title="Random Forest Feature Importance",
        output_name="random_forest_feature_importance.png",
    )

    plot_feature_importance(
        importances=xgb_importances,
        chart_title="XGBoost Feature Importance",
        output_name="xgboost_feature_importance.png",
    )


def print_final_conclusion(comparison: pd.DataFrame) -> None:
    """Print a concise final model summary."""
    best_row = comparison.iloc[0]
    best_model = best_row["model"]
    print("\nFinal Conclusion")
    print("-" * 60)
    print(
        f"The best model is {best_model} with ROC-AUC = {best_row['roc_auc']:.4f}, "
        f"Accuracy = {best_row['accuracy']:.4f}, Precision = {best_row['precision']:.4f}, "
        f"Recall = {best_row['recall']:.4f}, and F1-score = {best_row['f1_score']:.4f}."
    )
    print(
        f"TimeSeriesSplit cross-validation ROC-AUC: "
        f"{best_row['cv_roc_auc_mean']:.4f} +/- {best_row['cv_roc_auc_std']:.4f}"
    )
    print(f"Processed dataset saved to: {DATASET_PATH}")
    print(f"Model comparison saved to: {COMPARISON_PATH}")
    print(f"Figures saved to: {FIGURES_DIR}")


def main() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    ensure_output_directories()

    raw_data = download_market_data(TICKER, START_DATE, END_DATE)
    processed_data = engineer_features(raw_data)
    save_processed_dataset(processed_data)
    run_eda(raw_data, processed_data)

    X_train, X_test, y_train, y_test = split_data(processed_data)
    comparison, evaluation_cache, predictions = evaluate_models(X_train, X_test, y_train, y_test)

    feature_names = get_feature_columns()
    save_test_predictions(predictions)
    plot_roc_curves(y_test, evaluation_cache)
    plot_confusion_matrices(evaluation_cache)
    save_feature_importance_plots(evaluation_cache, feature_names)
    print_final_conclusion(comparison)


if __name__ == "__main__":
    main()
