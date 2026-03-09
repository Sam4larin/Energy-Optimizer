"""
model.py

Loads the trained XGBoost model and generates 24-hour consumption
forecasts from recent historical data.

The forecast is built iteratively — each predicted hour becomes
the lag input for the next hour, simulating real-world inference
where future consumption is unknown.
"""

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Tuple

from src.features import (
    add_time_features,
    add_rolling_features,
    FEATURE_COLS,
    TARGET
)


# Constants

MODELS_DIR = Path(__file__).resolve().parent.parent / 'models'


# Model Loading

def load_model() -> Tuple:
    """
    Loads the trained XGBoost model and feature column list from disk.

    Returns:
        Tuple of (model, feature_columns) where model is the trained
        XGBRegressor and feature_columns is the ordered list of feature names.

    Raises:
        FileNotFoundError: If model files are not found in models/ directory.
    """
    model_path    = MODELS_DIR / 'xgb_model.pkl'
    features_path = MODELS_DIR / 'feature_columns.json'

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Run 03_model_training.ipynb first."
        )

    model = joblib.load(model_path)

    with open(features_path, 'r') as f:
        feature_cols = json.load(f)

    return model, feature_cols


def load_metadata() -> dict:
    """
    Loads model performance metadata saved during training.

    Returns:
        Dictionary containing model parameters and evaluation metrics.
    """
    metadata_path = MODELS_DIR / 'model_metadata.json'

    with open(metadata_path, 'r') as f:
        return json.load(f)


# Forecasting

def make_forecast(
    recent_data: pd.DataFrame,
    forecast_date: pd.Timestamp,
    model,
    feature_cols: list,
    forecast_hours: int = 24
) -> pd.DataFrame:
    """
    Generates an hourly consumption forecast for the next N hours.

    The forecast is built iteratively:
    1. For hour 1: use real historical lags
    2. For hour 2+: use predicted values as lag inputs
    This simulates real-world inference where future values are unknown.

    Args:
        recent_data:    DataFrame of recent hourly consumption with all
                        original columns. Must contain at least 168 rows
                        of history before forecast_date.
        forecast_date:  The start datetime for the forecast (midnight of
                        the day to forecast).
        model:          Trained XGBRegressor loaded via load_model().
        feature_cols:   Ordered feature column list from load_model().
        forecast_hours: Number of hours to forecast. Default 24.

    Returns:
        DataFrame with columns ['ds', 'yhat'] where ds is the datetime
        and yhat is the predicted consumption in kW.
    """
    # Build the history buffer — we need 168 rows minimum for lag_168h
    history = recent_data[TARGET].copy()

    predictions = []
    forecast_index = pd.date_range(
        start=forecast_date,
        periods=forecast_hours,
        freq='1h'
    )

    for i, ts in enumerate(forecast_index):
        # Build a single-row DataFrame for this forecast hour
        row = pd.DataFrame(index=[ts])
        row[TARGET] = np.nan  # unknown — what we're predicting

        # Time features
        row['hour']        = ts.hour
        row['day_of_week'] = ts.dayofweek
        row['month']       = ts.month
        row['day_of_year'] = ts.dayofyear
        row['is_weekend']  = int(ts.dayofweek >= 5)
        row['is_nighttime'] = int(ts.hour >= 22 or ts.hour <= 5)

        row['hour_sin'] = np.sin(2 * np.pi * ts.hour / 24)
        row['hour_cos'] = np.cos(2 * np.pi * ts.hour / 24)
        row['day_sin']  = np.sin(2 * np.pi * ts.dayofweek / 7)
        row['day_cos']  = np.cos(2 * np.pi * ts.dayofweek / 7)

        # Lag features
        # Look back into history buffer (which includes prior predictions)
        def get_lag(n: int) -> float:
            """Retrieve value n hours before current forecast hour."""
            lag_ts = ts - pd.Timedelta(hours=n)
            if lag_ts in history.index:
                return history[lag_ts]
            return history.iloc[-1]  # fallback to most recent value

        row['lag_1h']   = get_lag(1)
        row['lag_2h']   = get_lag(2)
        row['lag_3h']   = get_lag(3)
        row['lag_24h']  = get_lag(24)
        row['lag_48h']  = get_lag(48)
        row['lag_168h'] = get_lag(168)

        # Rolling features
        recent_values = [
            history[ts - pd.Timedelta(hours=n)]
            if (ts - pd.Timedelta(hours=n)) in history.index
            else history.iloc[-1]
            for n in range(1, 25)
        ]

        row['rolling_mean_3h']  = np.mean(recent_values[:3])
        row['rolling_mean_24h'] = np.mean(recent_values[:24])
        row['rolling_std_24h']  = np.std(recent_values[:24])
        row['rolling_max_24h']  = np.max(recent_values[:24])

        # Weather features
        weather_cols = [
            'temperature', 'humidity', 'windSpeed', 'cloudCover',
            'precipIntensity', 'dewPoint', 'pressure', 'visibility',
            'apparentTemperature', 'windBearing', 'precipProbability'
        ]
        for col in weather_cols:
            if col in recent_data.columns:
                # Find the most recent weather reading before this hour
                past_weather = recent_data.loc[
                    recent_data.index < ts, col
                ]
                row[col] = past_weather.iloc[-1] if len(past_weather) > 0 else 0.0
            else:
                row[col] = 0.0

        # Predict
        X_row = row[feature_cols]
        pred  = float(model.predict(X_row)[0])
        pred  = max(0.0, pred)  # consumption cannot be negative

        predictions.append({'ds': ts, 'yhat': pred})

        # Add prediction to history so future lags can use it
        history[ts] = pred

    forecast_df = pd.DataFrame(predictions).set_index('ds')
    return forecast_df