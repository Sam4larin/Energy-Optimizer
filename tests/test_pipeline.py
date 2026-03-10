"""
Pytest tests for the Energy Optimizer pipeline.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features import FEATURE_COLS, TARGET, build_features, get_feature_matrix
from src.model import load_model, make_forecast
from src.optimizer import (
    APPLIANCE_LIBRARY,
    estimate_savings,
    find_optimal_window,
    generate_recommendations,
    get_daily_summary,
    get_flexible_appliances,
)


@pytest.fixture(scope="module")
def hourly_df():
    path = ROOT / "data" / "processed" / "hourly_data.csv"
    return pd.read_csv(path, index_col="time", parse_dates=True)


@pytest.fixture(scope="module")
def feature_matrix(hourly_df):
    return build_features(hourly_df)


@pytest.fixture(scope="module")
def model_and_cols():
    return load_model()


@pytest.fixture(scope="module")
def sample_forecast(hourly_df, model_and_cols):
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


class TestFeatureEngineering:
    def test_feature_matrix_shape(self, feature_matrix):
        X, _ = get_feature_matrix(feature_matrix)
        assert X.shape[1] == 31

    def test_all_feature_cols_present(self, feature_matrix):
        missing = [c for c in FEATURE_COLS if c not in feature_matrix.columns]
        assert not missing

    def test_no_nulls_in_features(self, feature_matrix):
        null_counts = feature_matrix[FEATURE_COLS].isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        assert len(cols_with_nulls) == 0

    def test_cyclical_features_bounded(self, feature_matrix):
        for col in ["hour_sin", "hour_cos", "day_sin", "day_cos"]:
            assert feature_matrix[col].between(-1, 1).all()

    def test_target_column_positive(self, feature_matrix):
        assert (feature_matrix[TARGET] >= 0).all()


class TestModel:
    def test_model_loads(self, model_and_cols):
        model, feature_cols = model_and_cols
        assert model is not None
        assert len(feature_cols) == 31

    def test_forecast_shape(self, sample_forecast):
        assert sample_forecast.shape[0] == 24

    def test_forecast_column_exists(self, sample_forecast):
        assert "yhat" in sample_forecast.columns

    def test_forecast_values_positive(self, sample_forecast):
        assert (sample_forecast["yhat"] >= 0).all()

    def test_forecast_values_plausible(self, sample_forecast):
        assert sample_forecast["yhat"].max() <= 10.0


class TestOptimizer:
    @pytest.fixture(scope="class")
    def flexible_appliances(self):
        names = [name for name, info in APPLIANCE_LIBRARY.items() if info["flexible"]]
        return get_flexible_appliances(names)

    def test_recommendations_generated(self, sample_forecast, flexible_appliances):
        recs = generate_recommendations(sample_forecast, flexible_appliances)
        assert len(recs) == len(flexible_appliances)

    def test_recommendations_have_required_keys(self, sample_forecast, flexible_appliances):
        required_keys = {
            "appliance", "message", "avoid_message",
            "saving", "recommended_time", "peak_time"
        }
        recs = generate_recommendations(sample_forecast, flexible_appliances)
        for rec in recs:
            assert not (required_keys - set(rec.keys()))

    def test_staggered_windows(self, sample_forecast, flexible_appliances):
        recs = generate_recommendations(sample_forecast, flexible_appliances)
        times = [rec["recommended_time"] for rec in recs]
        assert len(times) == len(set(times))

    def test_optimal_window_within_bounds(self, sample_forecast):
        hour = find_optimal_window(sample_forecast, run_hours=1.0)
        assert 0 <= hour <= 23

    def test_savings_non_negative_at_trough(self, sample_forecast):
        yhat = sample_forecast["yhat"].values
        peak_hour = int(np.argmax(yhat))
        trough_hour = int(np.argmin(yhat))
        dishwasher = APPLIANCE_LIBRARY["Dishwasher"]
        saving = estimate_savings(
            dishwasher["wattage"],
            dishwasher["run_hours"],
            trough_hour,
            peak_hour,
            sample_forecast,
            peak_rate=20.0,
            off_peak_rate=10.0,
        )
        assert saving["saving"] >= 0

    def test_daily_summary_keys(self, sample_forecast):
        summary = get_daily_summary(sample_forecast)
        required = {
            "peak_hour", "peak_time", "peak_value",
            "trough_hour", "trough_time", "trough_value",
            "total_kwh", "mean_kwh"
        }
        assert not (required - set(summary.keys()))
