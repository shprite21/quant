# Market Regime Detection and Stock Trend Prediction Using Machine Learning

This project downloads NIFTY 50 (`^NSEI`) historical data from Yahoo Finance, engineers technical indicators and market regime features, trains three machine learning models, compares them using time-aware validation, and exports the dataset, charts, results, and an interactive dashboard.

## Project Files

- `market_regime_detection.py`: End-to-end pipeline for download, preprocessing, EDA, modeling, evaluation, and export.
- `dashboard.py`: Streamlit dashboard for interactive exploration of prices, regimes, indicators, and model diagnostics.
- `launch_dashboard.py`: Lightweight launcher that starts the dashboard using local project packages.
- `processed_dataset.csv`: Cleaned dataset with engineered features and target label.
- `comparison_results.csv`: Model comparison table sorted by ROC-AUC.
- `test_predictions.csv`: Chronological test-set predictions and probabilities for all models.
- `random_forest_feature_importance.csv`: Feature importance export for Random Forest.
- `xgboost_feature_importance.csv`: Feature importance export for XGBoost.
- `figures/`: Saved charts for EDA, ROC curves, confusion matrices, and feature importance.

## Models

- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

## Outputs

- Historical closing price chart
- Daily return histogram
- Price with regime overlays
- Correlation heatmap
- Market regime distribution chart
- Indicator panels for RSI, MACD, and rolling volatility
- Monthly return heatmap
- ROC curve comparison
- Confusion matrix chart
- Random Forest feature importance chart
- XGBoost feature importance chart

## Dashboard

Run the training pipeline first if you want fresh data and metrics:

```powershell
python market_regime_detection.py
```

Then launch the interactive dashboard:

```powershell
python launch_dashboard.py
```

The dashboard includes:

- Date-range filtering
- Interactive price, SMA, and volume view
- Regime distribution and indicator analysis
- Model comparison, ROC curves, confusion matrix, and feature importance
- Download access for the exported CSV artifacts
