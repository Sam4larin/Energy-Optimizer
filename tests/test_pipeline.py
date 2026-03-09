"""
Pytest tests for the Energy Optimizer pipeline.
Covers feature engineering, model loading, and optimizer logic.
"""

import sys
import json
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

# Ensure src/ is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features import build_features, get_feature_matrix, FEATURE_COLS, TARGET
from src.model import load_model, make_forecast
from src.optimizer import (
    generate_recommendations,
    get_daily_summary,
    find_optimal_window,
    estimate_savings,
    APPLIANCES,
)


# Fixtures

@pytest.fixture(scope="module")
def hourly_df():
    """Load the processed hourly data once for all tests."""
    path = ROOT / "data" / "processed" / "hourly_data.csv"
    df = pd.read_csv(path, index_col="time", parse_dates=True)
    return df


@pytest.fixture(scope="module")
def feature_matrix(hourly_df):
    """Build feature matrix from hourly data."""
    return build_features(hourly_df)


@pytest.fixture(scope="module")
def model_and_cols():
    """Load model and feature columns once for all tests."""
    return load_model()


@pytest.fixture(scope="module")
def sample_forecast(hourly_df, model_and_cols):
    """Generate a 24-hour forecast for the last date in the dataset."""
    model, feature_cols = model_and_cols
    forecast_date = hourly_df.index.max().normalize()
    recent_data = hourly_df[hourly_df.index < forecast_date].tail(200)
    return make_forecast(
        recent_data=recent_data,
        forecast_date=forecast_date,
        model=model,
        feature_cols=feature_cols,
        forecast_hours=24,
    )


# Feature engineering tests

class TestFeatureEngineering:

    def test_feature_matrix_shape(self, feature_matrix):
        """Feature matrix must have exactly 31 columns."""
        X, y = get_feature_matrix(feature_matrix)
        assert X.shape[1] == 31, (
            f"Expected 31 features, got {X.shape[1]}"
        )

    def test_all_feature_cols_present(self, feature_matrix):
        """Every column in FEATURE_COLS must exist in the feature matrix."""
        missing = [c for c in FEATURE_COLS if c not in feature_matrix.columns]
        assert len(missing) == 0, f"Missing feature columns: {missing}"

    def test_no_nulls_in_features(self, feature_matrix):
        """Feature matrix must have zero null values."""
        null_counts = feature_matrix[FEATURE_COLS].isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        assert len(cols_with_nulls) == 0, (
            f"Null values found in: {cols_with_nulls.to_dict()}"
        )

    def test_cyclical_features_bounded(self, feature_matrix):
        """Sine and cosine features must stay within [-1, 1]."""
        for col in ["hour_sin", "hour_cos", "day_sin", "day_cos"]:
            assert feature_matrix[col].between(-1, 1).all(), (
                f"{col} contains values outside [-1, 1]"
            )

    def test_target_column_positive(self, feature_matrix):
        """Target (use [kW]) must be non-negative throughout."""
        assert (feature_matrix[TARGET] >= 0).all(), (
            "Negative consumption values found in target column"
        )


# Model tests

class TestModel:

    def test_model_loads(self, model_and_cols):
        """Model and feature columns must load without error."""
        model, feature_cols = model_and_cols
        assert model is not None
        assert len(feature_cols) == 31

    def test_forecast_shape(self, sample_forecast):
        """Forecast must return exactly 24 rows."""
        assert sample_forecast.shape[0] == 24, (
            f"Expected 24 forecast rows, got {sample_forecast.shape[0]}"
        )

    def test_forecast_column_exists(self, sample_forecast):
        """Forecast DataFrame must contain a 'yhat' column."""
        assert "yhat" in sample_forecast.columns

    def test_forecast_values_positive(self, sample_forecast):
        """All forecast values must be non-negative."""
        assert (sample_forecast["yhat"] >= 0).all(), (
            "Negative forecast values detected"
        )

    def test_forecast_values_plausible(self, sample_forecast):
        """Forecast values must be within a plausible household range (0–10 kW)."""
        assert sample_forecast["yhat"].max() <= 10.0, (
            f"Forecast max {sample_forecast['yhat'].max():.2f} kW exceeds 10 kW"
        )


# Optimizer tests

class TestOptimizer:

    def test_recommendations_generated(self, sample_forecast):
        """Recommendations must be generated for all flexible appliances."""
        appliances = [a for a, v in APPLIANCES.items() if v["flexible"]]
        recs = generate_recommendations(sample_forecast, appliances)
        assert len(recs) == len(appliances), (
            f"Expected {len(appliances)} recommendations, got {len(recs)}"
        )

    def test_recommendations_have_required_keys(self, sample_forecast):
        """Each recommendation dict must contain all required keys."""
        required_keys = {
            "appliance", "message", "avoid_message",
            "saving", "recommended_time", "peak_time"
        }
        appliances = [a for a, v in APPLIANCES.items() if v["flexible"]]
        recs = generate_recommendations(sample_forecast, appliances)
        for rec in recs:
            missing = required_keys - set(rec.keys())
            assert len(missing) == 0, (
                f"Recommendation for {rec.get('appliance')} missing keys: {missing}"
            )

    def test_staggered_windows(self, sample_forecast):
        """No two appliances should be assigned the exact same start hour."""
        appliances = [a for a, v in APPLIANCES.items() if v["flexible"]]
        recs = generate_recommendations(sample_forecast, appliances)
        times = [rec["recommended_time"] for rec in recs]
        assert len(times) == len(set(times)), (
            f"Duplicate recommendation times found: {times}"
        )

    def test_optimal_window_within_bounds(self, sample_forecast):
        """Optimal window must be a valid hour between 0 and 23."""
        hour = find_optimal_window(sample_forecast, run_hours=1.0)
        assert 0 <= hour <= 23, f"Optimal hour {hour} is out of bounds"

    def test_savings_non_negative_at_trough(self, sample_forecast):
        """Running at the trough vs the peak should never produce negative savings."""
        yhat        = sample_forecast["yhat"].values
        peak_hour   = int(np.argmax(yhat))
        trough_hour = int(np.argmin(yhat))
        saving = estimate_savings(
            "Dishwasher", trough_hour, peak_hour, sample_forecast
        )
        assert saving >= 0, (
            f"Negative savings ({saving}) when running at trough vs peak"
        )

    def test_daily_summary_keys(self, sample_forecast):
        """Daily summary must contain all expected keys."""
        summary = get_daily_summary(sample_forecast)
        required = {
            "peak_hour", "peak_time", "peak_value",
            "trough_hour", "trough_time", "trough_value",
            "total_kwh", "mean_kwh"
        }
        missing = required - set(summary.keys())
        assert len(missing) == 0, f"Daily summary missing keys: {missing}"