"""Interactive dashboard for market regime detection and trend prediction."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_PACKAGES = PROJECT_ROOT / ".python_packages"

if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from market_regime_detection import (
    COMPARISON_PATH,
    DATASET_PATH,
    PREDICTIONS_PATH,
    RF_IMPORTANCE_PATH,
    XGB_IMPORTANCE_PATH,
    REGIME_COLORS,
    main as run_pipeline,
)

MODEL_COLUMN_MAP = {
    "Logistic Regression": {
        "prediction": "logistic_regression_prediction",
        "probability": "logistic_regression_probability",
    },
    "Random Forest": {
        "prediction": "random_forest_prediction",
        "probability": "random_forest_probability",
    },
    "XGBoost": {
        "prediction": "xgboost_prediction",
        "probability": "xgboost_probability",
    },
}

MODEL_IMPORTANCE_PATHS = {
    "Random Forest": RF_IMPORTANCE_PATH,
    "XGBoost": XGB_IMPORTANCE_PATH,
}


def empty_message_figure(title: str, message: str) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 16},
    )
    figure.update_layout(
        title=title,
        height=420,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return figure


def artifacts_available() -> bool:
    required_files = [
        DATASET_PATH,
        COMPARISON_PATH,
        PREDICTIONS_PATH,
        RF_IMPORTANCE_PATH,
        XGB_IMPORTANCE_PATH,
    ]
    return all(path.exists() for path in required_files)


@st.cache_data(show_spinner=False)
def load_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    processed_data = pd.read_csv(DATASET_PATH, parse_dates=["date"])
    comparison = pd.read_csv(COMPARISON_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH, parse_dates=["date"])
    importances = {
        model_name: pd.read_csv(path)
        for model_name, path in MODEL_IMPORTANCE_PATHS.items()
    }

    processed_data = processed_data.sort_values("date").reset_index(drop=True)
    predictions = predictions.sort_values("date").reset_index(drop=True)
    return processed_data, comparison, predictions, importances


def refresh_pipeline() -> None:
    with st.spinner("Refreshing data, retraining models, and regenerating artifacts..."):
        run_pipeline()
    st.cache_data.clear()
    st.rerun()


def render_sidebar(processed_data: pd.DataFrame, comparison: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp, str]:
    with st.sidebar:
        st.header("Controls")
        st.caption("Use the filters below to explore market behavior and model performance.")

        if st.button("Refresh Pipeline", use_container_width=True):
            refresh_pipeline()

        min_date = processed_data["date"].min().date()
        max_date = processed_data["date"].max().date()
        date_range = st.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        if len(date_range) != 2:
            start_date, end_date = min_date, max_date
        else:
            start_date, end_date = date_range

        best_model = comparison.iloc[0]["model"]
        selected_model = st.selectbox(
            "Model Diagnostics",
            options=list(MODEL_COLUMN_MAP.keys()),
            index=list(MODEL_COLUMN_MAP.keys()).index(best_model),
        )

        st.markdown("---")
        st.markdown(
            "Dashboard artifacts are sourced from the training pipeline. "
            "Refreshing the pipeline downloads the latest Yahoo Finance data."
        )

    return pd.Timestamp(start_date), pd.Timestamp(end_date), selected_model


def filter_data(
    processed_data: pd.DataFrame,
    predictions: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    filtered_data = processed_data[
        (processed_data["date"] >= start_date) & (processed_data["date"] <= end_date)
    ].copy()
    filtered_predictions = predictions[
        (predictions["date"] >= start_date) & (predictions["date"] <= end_date)
    ].copy()
    return filtered_data, filtered_predictions


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def render_header(filtered_data: pd.DataFrame, comparison: pd.DataFrame) -> None:
    best_row = comparison.iloc[0]
    latest_row = filtered_data.iloc[-1]
    cumulative_return = (latest_row["close"] / filtered_data.iloc[0]["close"]) - 1
    annualized_volatility = filtered_data["daily_return"].std() * np.sqrt(252)
    annualized_volatility = 0.0 if pd.isna(annualized_volatility) else annualized_volatility
    dominant_regime = filtered_data["regime"].mode().iloc[0].title()

    st.title("Market Regime Detection Dashboard")
    st.caption("Interactive view of NIFTY 50 regimes, technical indicators, and model diagnostics.")

    metric_columns = st.columns(5)
    metric_columns[0].metric("Latest Close", f"{latest_row['close']:.2f}")
    metric_columns[1].metric("Window Return", format_percent(cumulative_return))
    metric_columns[2].metric("Annualized Volatility", format_percent(annualized_volatility))
    metric_columns[3].metric("Dominant Regime", dominant_regime)
    metric_columns[4].metric("Best ROC-AUC", f"{best_row['model']} ({best_row['roc_auc']:.3f})")


def create_price_figure(filtered_data: pd.DataFrame) -> go.Figure:
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.05,
    )

    figure.add_trace(
        go.Scatter(
            x=filtered_data["date"],
            y=filtered_data["close"],
            mode="lines",
            name="Close",
            line={"color": "#111827", "width": 2.4},
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=filtered_data["date"],
            y=filtered_data["sma_20"],
            mode="lines",
            name="SMA 20",
            line={"color": "#2563eb", "width": 1.6},
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=filtered_data["date"],
            y=filtered_data["sma_50"],
            mode="lines",
            name="SMA 50",
            line={"color": "#7c3aed", "width": 1.6},
        ),
        row=1,
        col=1,
    )

    for regime in ["bullish", "sideways", "bearish"]:
        regime_slice = filtered_data[filtered_data["regime"] == regime]
        figure.add_trace(
            go.Scatter(
                x=regime_slice["date"],
                y=regime_slice["close"],
                mode="markers",
                name=f"{regime.title()} Regime",
                marker={"size": 5, "color": REGIME_COLORS[regime], "opacity": 0.75},
            ),
            row=1,
            col=1,
        )

    figure.add_trace(
        go.Bar(
            x=filtered_data["date"],
            y=filtered_data["volume"],
            name="Volume",
            marker={"color": "#94a3b8"},
            opacity=0.6,
        ),
        row=2,
        col=1,
    )

    figure.update_layout(
        title="Price, Trend, and Volume",
        height=650,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
    )
    figure.update_yaxes(title_text="Close", row=1, col=1)
    figure.update_yaxes(title_text="Volume", row=2, col=1)
    return figure


def create_return_distribution(filtered_data: pd.DataFrame) -> go.Figure:
    figure = px.histogram(
        filtered_data,
        x="daily_return",
        nbins=50,
        marginal="box",
        title="Daily Return Distribution",
        color_discrete_sequence=["#0f766e"],
    )
    figure.update_layout(height=420, margin={"l": 20, "r": 20, "t": 60, "b": 20})
    figure.update_xaxes(title="Daily Return")
    figure.update_yaxes(title="Count")
    return figure


def create_monthly_heatmap(filtered_data: pd.DataFrame) -> go.Figure:
    monthly_close = filtered_data.set_index("date")["close"].resample("ME").last()
    monthly_returns = monthly_close.pct_change().dropna()
    if monthly_returns.empty:
        return empty_message_figure(
            "Monthly Return Heatmap (%)",
            "Select a wider date range to compute monthly returns.",
        )

    monthly_frame = monthly_returns.to_frame(name="monthly_return")
    monthly_frame["year"] = monthly_frame.index.year
    monthly_frame["month"] = monthly_frame.index.strftime("%b")

    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot = monthly_frame.pivot(index="year", columns="month", values="monthly_return")
    pivot = pivot.reindex(columns=month_order)
    if pivot.empty:
        return empty_message_figure(
            "Monthly Return Heatmap (%)",
            "Monthly return data is not available for this window.",
        )

    figure = px.imshow(
        pivot * 100,
        color_continuous_scale="RdYlGn",
        aspect="auto",
        title="Monthly Return Heatmap (%)",
        text_auto=".1f",
    )
    figure.update_layout(height=420, margin={"l": 20, "r": 20, "t": 60, "b": 20})
    return figure


def create_regime_distribution(filtered_data: pd.DataFrame) -> go.Figure:
    distribution = (
        filtered_data["regime"]
        .value_counts()
        .reindex(["bullish", "sideways", "bearish"], fill_value=0)
        .reset_index()
    )
    distribution.columns = ["regime", "count"]

    figure = px.bar(
        distribution,
        x="regime",
        y="count",
        color="regime",
        title="Regime Distribution",
        color_discrete_map=REGIME_COLORS,
    )
    figure.update_layout(height=420, showlegend=False, margin={"l": 20, "r": 20, "t": 60, "b": 20})
    return figure


def create_indicator_figure(filtered_data: pd.DataFrame) -> go.Figure:
    macd_histogram = filtered_data["macd"] - filtered_data["macd_signal"]
    volatility_median = filtered_data["rolling_volatility_20"].median()
    bar_colors = np.where(macd_histogram >= 0, "#16a34a", "#dc2626")

    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("RSI (14)", "MACD vs Signal", "20-Day Rolling Volatility"),
    )

    figure.add_trace(
        go.Scatter(
            x=filtered_data["date"],
            y=filtered_data["rsi_14"],
            mode="lines",
            name="RSI 14",
            line={"color": "#ea580c", "width": 1.8},
        ),
        row=1,
        col=1,
    )
    figure.add_hline(y=70, line_dash="dash", line_color="#dc2626", row=1, col=1)
    figure.add_hline(y=30, line_dash="dash", line_color="#16a34a", row=1, col=1)

    figure.add_trace(
        go.Bar(
            x=filtered_data["date"],
            y=macd_histogram,
            name="MACD Histogram",
            marker={"color": bar_colors},
            opacity=0.35,
        ),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=filtered_data["date"],
            y=filtered_data["macd"],
            mode="lines",
            name="MACD",
            line={"color": "#2563eb", "width": 1.6},
        ),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=filtered_data["date"],
            y=filtered_data["macd_signal"],
            mode="lines",
            name="MACD Signal",
            line={"color": "#7c3aed", "width": 1.6},
        ),
        row=2,
        col=1,
    )

    figure.add_trace(
        go.Scatter(
            x=filtered_data["date"],
            y=filtered_data["rolling_volatility_20"],
            mode="lines",
            name="Rolling Volatility",
            line={"color": "#92400e", "width": 1.8},
        ),
        row=3,
        col=1,
    )
    figure.add_hline(y=volatility_median, line_dash="dash", line_color="#64748b", row=3, col=1)

    figure.update_layout(height=850, margin={"l": 20, "r": 20, "t": 80, "b": 20})
    return figure


def create_comparison_chart(comparison: pd.DataFrame) -> go.Figure:
    metric_frame = comparison.melt(
        id_vars="model",
        value_vars=["roc_auc", "accuracy", "f1_score", "precision", "recall", "cv_roc_auc_mean"],
        var_name="metric",
        value_name="score",
    )

    label_map = {
        "roc_auc": "Test ROC-AUC",
        "accuracy": "Accuracy",
        "f1_score": "F1 Score",
        "precision": "Precision",
        "recall": "Recall",
        "cv_roc_auc_mean": "CV ROC-AUC",
    }
    metric_frame["metric"] = metric_frame["metric"].map(label_map)

    figure = px.bar(
        metric_frame,
        x="metric",
        y="score",
        color="model",
        barmode="group",
        title="Model Comparison",
        color_discrete_sequence=["#2563eb", "#0f766e", "#7c3aed"],
    )
    figure.update_layout(height=430, margin={"l": 20, "r": 20, "t": 60, "b": 20})
    figure.update_yaxes(range=[0, 1])
    return figure


def create_roc_chart(predictions: pd.DataFrame) -> go.Figure:
    actual = predictions["actual"]
    if actual.nunique() < 2:
        return empty_message_figure("ROC Curves", "Select a wider test window with both classes present.")

    figure = go.Figure()

    for model_name, columns in MODEL_COLUMN_MAP.items():
        probability_column = columns["probability"]
        if probability_column not in predictions:
            continue
        fpr, tpr, _ = roc_curve(actual, predictions[probability_column])
        auc_score = roc_auc_score(actual, predictions[probability_column])
        figure.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{model_name} (AUC = {auc_score:.3f})",
            )
        )

    figure.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Baseline",
            line={"dash": "dash", "color": "#94a3b8"},
        )
    )
    figure.update_layout(
        title="ROC Curves",
        height=420,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
    )
    return figure


def create_probability_chart(predictions: pd.DataFrame, model_name: str) -> go.Figure:
    probability_column = MODEL_COLUMN_MAP[model_name]["probability"]

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=predictions["date"],
            y=predictions[probability_column],
            mode="lines",
            name=f"{model_name} Probability",
            line={"color": "#2563eb", "width": 2},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=predictions["date"],
            y=predictions["actual"],
            mode="lines",
            name="Actual Direction",
            line={"color": "#111827", "width": 1.5, "dash": "dot"},
        )
    )
    figure.add_hline(y=0.5, line_dash="dash", line_color="#dc2626")
    figure.update_layout(
        title=f"{model_name} Test-Set Probabilities vs Actual Direction",
        height=420,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        yaxis_title="Probability / Actual Label",
    )
    return figure


def create_confusion_chart(predictions: pd.DataFrame, model_name: str) -> go.Figure:
    prediction_column = MODEL_COLUMN_MAP[model_name]["prediction"]
    matrix = confusion_matrix(predictions["actual"], predictions[prediction_column], labels=[0, 1])

    figure = go.Figure(
        data=
        [
            go.Heatmap(
                z=matrix,
                x=["Predicted Down", "Predicted Up"],
                y=["Actual Down", "Actual Up"],
                text=matrix,
                texttemplate="%{text}",
                colorscale="Blues",
            )
        ]
    )
    figure.update_layout(
        title=f"{model_name} Confusion Matrix",
        height=420,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )
    return figure


def create_feature_importance_chart(importance_frame: pd.DataFrame, model_name: str) -> go.Figure:
    top_features = importance_frame.sort_values("importance", ascending=False).head(12).sort_values("importance")
    figure = px.bar(
        top_features,
        x="importance",
        y="feature",
        orientation="h",
        title=f"{model_name} Feature Importance",
        color="importance",
        color_continuous_scale="Viridis",
    )
    figure.update_layout(height=420, margin={"l": 20, "r": 20, "t": 60, "b": 20}, coloraxis_showscale=False)
    return figure


def render_overview_tab(filtered_data: pd.DataFrame) -> None:
    left_column, right_column = st.columns([1.9, 1.1])
    with left_column:
        st.plotly_chart(create_price_figure(filtered_data), use_container_width=True)
    with right_column:
        st.plotly_chart(create_regime_distribution(filtered_data), use_container_width=True)
        st.plotly_chart(create_return_distribution(filtered_data), use_container_width=True)

    st.plotly_chart(create_monthly_heatmap(filtered_data), use_container_width=True)


def render_regime_tab(filtered_data: pd.DataFrame) -> None:
    st.plotly_chart(create_indicator_figure(filtered_data), use_container_width=True)

    regime_return_table = (
        filtered_data.groupby("regime")["daily_return"]
        .agg(["count", "mean", "std"])
        .reindex(["bullish", "sideways", "bearish"])
        .rename(columns={"count": "days", "mean": "avg_return", "std": "return_std"})
    )
    regime_return_table["avg_return"] = regime_return_table["avg_return"].map(lambda value: f"{value * 100:.3f}%")
    regime_return_table["return_std"] = regime_return_table["return_std"].map(lambda value: f"{value * 100:.3f}%")
    st.dataframe(regime_return_table, use_container_width=True)


def render_model_tab(
    comparison: pd.DataFrame,
    filtered_predictions: pd.DataFrame,
    importances: dict[str, pd.DataFrame],
    selected_model: str,
) -> None:
    st.plotly_chart(create_comparison_chart(comparison), use_container_width=True)

    if filtered_predictions.empty:
        st.info("The selected date window does not include any test-set predictions yet.")
        return

    left_column, right_column = st.columns(2)
    with left_column:
        st.plotly_chart(create_roc_chart(filtered_predictions), use_container_width=True)
    with right_column:
        st.plotly_chart(create_confusion_chart(filtered_predictions, selected_model), use_container_width=True)

    left_column, right_column = st.columns(2)
    with left_column:
        st.plotly_chart(create_probability_chart(filtered_predictions, selected_model), use_container_width=True)
    with right_column:
        tree_model = "Random Forest" if selected_model == "Logistic Regression" else selected_model
        st.plotly_chart(
            create_feature_importance_chart(importances[tree_model], tree_model),
            use_container_width=True,
        )


def render_data_tab(
    filtered_data: pd.DataFrame,
    comparison: pd.DataFrame,
    filtered_predictions: pd.DataFrame,
) -> None:
    st.subheader("Processed Dataset Preview")
    st.dataframe(filtered_data.tail(50), use_container_width=True)
    st.download_button(
        "Download Processed Dataset",
        DATASET_PATH.read_bytes(),
        file_name=DATASET_PATH.name,
        mime="text/csv",
    )

    st.subheader("Model Comparison")
    st.dataframe(comparison, use_container_width=True)
    st.download_button(
        "Download Comparison Results",
        COMPARISON_PATH.read_bytes(),
        file_name=COMPARISON_PATH.name,
        mime="text/csv",
    )

    st.subheader("Test Predictions")
    st.dataframe(filtered_predictions.tail(50), use_container_width=True)
    st.download_button(
        "Download Test Predictions",
        PREDICTIONS_PATH.read_bytes(),
        file_name=PREDICTIONS_PATH.name,
        mime="text/csv",
    )


def main() -> None:
    st.set_page_config(
        page_title="Market Regime Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if not artifacts_available():
        st.title("Market Regime Detection Dashboard")
        st.warning("Artifacts are missing. Generate them first to launch the dashboard.")
        if st.button("Generate Artifacts", use_container_width=True):
            refresh_pipeline()
        st.stop()

    processed_data, comparison, predictions, importances = load_artifacts()
    start_date, end_date, selected_model = render_sidebar(processed_data, comparison)
    filtered_data, filtered_predictions = filter_data(processed_data, predictions, start_date, end_date)

    if filtered_data.empty:
        st.error("No records found for the selected date range.")
        st.stop()

    render_header(filtered_data, comparison)

    overview_tab, regime_tab, model_tab, data_tab = st.tabs(
        ["Overview", "Regimes & Indicators", "Model Diagnostics", "Data Explorer"]
    )

    with overview_tab:
        render_overview_tab(filtered_data)

    with regime_tab:
        render_regime_tab(filtered_data)

    with model_tab:
        render_model_tab(comparison, filtered_predictions, importances, selected_model)

    with data_tab:
        render_data_tab(filtered_data, comparison, filtered_predictions)


if __name__ == "__main__":
    main()
