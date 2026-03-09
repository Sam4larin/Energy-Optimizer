"""
features.py

Transforms raw hourly energy data into the feature matrix
used for model training and inference.

All feature engineering logic lives here. The dashboard and
model training notebook both import from this file — no
duplicated logic anywhere in the project.
"""

import pandas as pd
import numpy as np
from typing import List


# Constants

TARGET = 'use [kW]'

FEATURE_COLS: List[str] = [
    # Time features
    'hour', 'day_of_week', 'month', 'day_of_year',
    'is_weekend', 'is_nighttime',
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
    # Lag features
    'lag_1h', 'lag_2h', 'lag_3h',
    'lag_24h', 'lag_48h', 'lag_168h',
    # Rolling features
    'rolling_mean_3h', 'rolling_mean_24h',
    'rolling_std_24h', 'rolling_max_24h',
    # Weather features
    'temperature', 'humidity', 'windSpeed',
    'cloudCover', 'precipIntensity', 'dewPoint',
    'pressure', 'visibility', 'apparentTemperature',
    'windBearing', 'precipProbability'
]


# Feature Engineering

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds calendar and cyclical time features derived from the datetime index.

    Cyclical encoding (sin/cos) ensures the model understands that
    hour 23 and hour 0 are adjacent, not 23 steps apart.

    Args:
        df: DataFrame with a DatetimeIndex.

    Returns:
        DataFrame with time feature columns added.
    """
    df = df.copy()

    df['hour']        = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month']       = df.index.month
    df['day_of_year'] = df.index.dayofyear
    df['is_weekend']  = (df.index.dayofweek >= 5).astype(int)
    df['is_nighttime'] = (
        (df.index.hour >= 22) | (df.index.hour <= 5)
    ).astype(int)

    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin']  = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos']  = np.cos(2 * np.pi * df['day_of_week'] / 7)

    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds lag features that look backward at past consumption values.

    Lag features encode household habit patterns — what happened at
    this hour yesterday and last week are the strongest predictors
    of what will happen now.

    Args:
        df: DataFrame with a TARGET column and DatetimeIndex.

    Returns:
        DataFrame with lag columns added. First 168 rows will contain
        NaN values — drop these before training.
    """
    df = df.copy()

    df['lag_1h']   = df[TARGET].shift(1)
    df['lag_2h']   = df[TARGET].shift(2)
    df['lag_3h']   = df[TARGET].shift(3)
    df['lag_24h']  = df[TARGET].shift(24)
    df['lag_48h']  = df[TARGET].shift(48)
    df['lag_168h'] = df[TARGET].shift(168)

    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds rolling window statistics capturing recent consumption trends.

    Rolling features answer: is the household currently in a
    high-consumption period? Is behaviour stable or volatile?

    Args:
        df: DataFrame with a TARGET column and DatetimeIndex.

    Returns:
        DataFrame with rolling statistic columns added.
    """
    df = df.copy()

    shifted = df[TARGET].shift(1)

    df['rolling_mean_3h']  = shifted.rolling(window=3,  min_periods=1).mean()
    df['rolling_mean_24h'] = shifted.rolling(window=24, min_periods=1).mean()
    df['rolling_std_24h']  = shifted.rolling(window=24, min_periods=1).std()
    df['rolling_max_24h']  = shifted.rolling(window=24, min_periods=1).max()

    return df


def build_features(df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Applies all feature transformations in the correct order and
    optionally drops warmup rows that lack sufficient lag history.

    Args:
        df:      Cleaned hourly DataFrame with TARGET column and DatetimeIndex.
        drop_na: If True, drops rows with NaN from lag warmup period.
                 Set to False during inference when building future rows.

    Returns:
        DataFrame with all 31 model features plus the target column.
    """
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)

    if drop_na:
        rows_before = len(df)
        df = df.dropna()
        rows_dropped = rows_before - len(df)
        if rows_dropped > 0:
            print(f"Dropped {rows_dropped} warmup rows (insufficient lag history)")

    return df


def get_feature_matrix(df: pd.DataFrame) -> tuple:
    """
    Returns X (features) and y (target) arrays ready for model training.

    Args:
        df: DataFrame that has already been through build_features().

    Returns:
        Tuple of (X, y) where X is the feature DataFrame and y is the
        target Series.
    """
    X = df[FEATURE_COLS]
    y = df[TARGET]
    return X, y