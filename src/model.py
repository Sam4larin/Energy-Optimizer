"""
src/model.py

Model loading, retraining, and forecast generation.

Supports two modes:
- Demo mode: loads the pre-trained XGBoost model from models/
- User mode: retrains XGBoost on the user's uploaded data
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Optional

from xgboost import XGBRegressor

from src.features import build_features, get_feature_matrix

# Paths
MODELS_DIR = Path(__file__).resolve().parent.parent / 'models'


# Demo mode — load pre-trained model

def load_model() -> Tuple[XGBRegressor, List[str]]:
    """
    Loads the pre-trained XGBoost model and feature columns.

    Returns:
        Tuple of (model, feature_cols).
    """
    model        = joblib.load(MODELS_DIR / 'xgb_model.pkl')
    feature_cols = json.loads(
        (MODELS_DIR / 'feature_columns.json').read_text()
    )
    return model, feature_cols


def load_metadata() -> Dict:
    """Loads model performance metadata."""
    return json.loads(
        (MODELS_DIR / 'model_metadata.json').read_text()
    )


# User mode — retrain on uploaded data

def retrain_on_user_data(
    hourly_df: pd.DataFrame,
    progress_callback=None,
) -> Tuple[XGBRegressor, List[str], Dict]:
    """
    Retrains XGBoost on the user's uploaded hourly meter data.

    Uses the same feature engineering pipeline as the demo model.
    Applies a temporal 80/20 train/test split.

    Args:
        hourly_df:          Normalised hourly DataFrame with 'use [kW]' column.
        progress_callback:  Optional callable(message: str) for UI updates.

    Returns:
        Tuple of:
            model:        Trained XGBRegressor
            feature_cols: List of feature column names
            metadata:     Dict with performance metrics
    """
    def log(msg: str):
        if progress_callback:
            progress_callback(msg)

    log("Building features from your data...")
    feature_df = build_features(hourly_df, drop_na=True)

    if len(feature_df) < 168:
        raise ValueError(
            "Not enough data after feature engineering. "
            "Please upload at least 3 weeks of hourly readings."
        )

    X, y = get_feature_matrix(feature_df)
    feature_cols = list(X.columns)

    # Temporal split — never shuffle time series data
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    log("Training XGBoost model on your data...")

    model = XGBRegressor(
        n_estimators=700,
        learning_rate=0.02,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    log("Evaluating model performance...")

    y_pred = model.predict(X_test)
    y_pred = np.maximum(y_pred, 0)

    # MAPE on meaningful hours only
    mask = y_test > 0.1
    if mask.sum() > 0:
        mape = float(
            np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
        )
    else:
        mape = None

    mae = float(np.mean(np.abs(y_test - y_pred)))

    # Naive baseline: predict same hour yesterday
    if len(y_test) > 24:
        y_baseline = y_test.shift(24).dropna()
        y_actual   = y_test[y_baseline.index]
        mask_b     = y_actual > 0.1
        if mask_b.sum() > 0:
            baseline_mape = float(
                np.mean(
                    np.abs((y_actual[mask_b] - y_baseline[mask_b]) / y_actual[mask_b])
                ) * 100
            )
        else:
            baseline_mape = None
    else:
        baseline_mape = None

    metadata = {
        'model_mape':    round(mape, 2) if mape else 'N/A',
        'baseline_mape': round(baseline_mape, 2) if baseline_mape else 'N/A',
        'mae':           round(mae, 4),
        'train_rows':    len(X_train),
        'test_rows':     len(X_test),
        'feature_count': len(feature_cols),
        'data_days':     (hourly_df.index.max() - hourly_df.index.min()).days,
        'source':        'user_upload',
    }

    log("Model ready.")

    return model, feature_cols, metadata


# Forecast generation

def make_forecast(
    recent_data: pd.DataFrame,
    forecast_date: pd.Timestamp,
    model: XGBRegressor,
    feature_cols: List[str],
    forecast_hours: int = 24,
    future_weather: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Generates an iterative 24-hour forecast.

    Uses the trained model to predict hour by hour, feeding each
    prediction back as a lag input for the next hour. This is how
    the model behaves in production — no ground truth available.

    Args:
        recent_data:    Recent hourly DataFrame with 'use [kW]' column.
                        Must contain at least 168 rows (1 week) before
                        forecast_date for lag features to work.
        forecast_date:  The date to forecast (midnight = start of day).
        model:          Trained XGBRegressor.
        feature_cols:   List of feature column names the model expects.
        forecast_hours: Number of hours to forecast (default 24).
        future_weather: Optional weather frame indexed by forecast timestamps.

    Returns:
        DataFrame with DatetimeIndex and 'yhat' column (predicted kW).
    """

    history = recent_data[['use [kW]']].copy()

    predictions = []
    timestamps  = []

    for h in range(forecast_hours):
        target_ts = forecast_date + pd.Timedelta(hours=h)

        # Build a single-row feature set for this hour
        # We need 168+ hours of history for lag_168h
        working = history.copy()

        # Add a placeholder row at target_ts so features can be computed
        placeholder = pd.DataFrame(
            {'use [kW]': [np.nan]},
            index=[target_ts]
        )
        working = pd.concat([working, placeholder])

        # Build features
        try:
            feature_df = build_features(working, drop_na=False)
        except Exception:
            # If feature building fails, use last known value
            predictions.append(history['use [kW]'].iloc[-1])
            timestamps.append(target_ts)
            continue

        # Get the row for this specific hour
        if target_ts not in feature_df.index:
            predictions.append(history['use [kW]'].iloc[-1])
            timestamps.append(target_ts)
            continue

        # Add any missing columns the model expects (e.g. weather cols)
        # as zeros — model will use time/lag features as primary signal
        for col in feature_cols:
            if col not in feature_df.columns:
                feature_df[col] = 0.0

        row = feature_df.loc[[target_ts], feature_cols]

        # Inject forecast-time weather when available.
        if future_weather is not None and target_ts in future_weather.index:
            weather_row = future_weather.loc[target_ts]
            if isinstance(weather_row, pd.DataFrame):
                weather_row = weather_row.iloc[0]
            for col in feature_cols:
                if col in weather_row.index:
                    row.loc[target_ts, col] = weather_row[col]

        row = row.ffill().bfill().fillna(0)

        # Predict
        yhat = float(model.predict(row)[0])
        yhat = max(yhat, 0)

        predictions.append(yhat)
        timestamps.append(target_ts)

        # Append prediction to history so next hour's lags are correct
        new_row = pd.DataFrame(
            {'use [kW]': [yhat]},
            index=[target_ts]
        )
        history = pd.concat([history, new_row])

    forecast = pd.DataFrame(
        {'yhat': predictions},
        index=pd.DatetimeIndex(timestamps, name='ds')
    )

    return forecast
