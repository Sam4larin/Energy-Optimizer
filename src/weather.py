"""
src/weather.py

Fetches historical and forecast weather from Open-Meteo.
Free, no API key required, works worldwide.

Weather columns match the original smart_home.csv feature names:
    temperature, apparentTemperature, humidity, windSpeed, windBearing,
    cloudCover, precipIntensity, precipProbability, pressure,
    dewPoint, visibility
"""

import requests
import pandas as pd
from typing import Tuple


# Open-Meteo variable → our feature column name
VARIABLE_MAP = {
    "temperature_2m":            "temperature",
    "apparent_temperature":      "apparentTemperature",
    "relative_humidity_2m":      "humidity",
    "wind_speed_10m":            "windSpeed",
    "wind_direction_10m":        "windBearing",
    "cloud_cover":               "cloudCover",
    "precipitation":             "precipIntensity",
    "precipitation_probability": "precipProbability",
    "surface_pressure":          "pressure",
    "dew_point_2m":              "dewPoint",
    "visibility":                "visibility",
}

HISTORICAL_VARS = [
    "temperature_2m", "apparent_temperature", "relative_humidity_2m",
    "wind_speed_10m", "wind_direction_10m", "cloud_cover",
    "precipitation", "surface_pressure", "dew_point_2m", "visibility",
]

FORECAST_VARS = HISTORICAL_VARS + ["precipitation_probability"]

WEATHER_COLS = list(VARIABLE_MAP.values())


def geocode(location: str) -> Tuple[float, float, str]:
    """
    Converts a city name or postcode to lat/lon using Open-Meteo geocoding.

    Returns:
        (latitude, longitude, resolved_display_name)

    Raises:
        ValueError if location cannot be found.
    """
    url  = "https://geocoding-api.open-meteo.com/v1/search"
    resp = requests.get(url, params={"name": location, "count": 1}, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if not data.get("results"):
        raise ValueError(
            f"Could not find '{location}'. "
            "Try a city name like 'Lagos' or 'London, UK'."
        )

    r    = data["results"][0]
    name = f"{r['name']}, {r.get('country', '')}"
    return float(r["latitude"]), float(r["longitude"]), name


def _normalise_units(df: pd.DataFrame) -> pd.DataFrame:
    """Converts Open-Meteo units to match the original dataset conventions."""
    if "humidity"          in df.columns:
        df["humidity"]          = df["humidity"] / 100.0
    if "cloudCover"        in df.columns:
        df["cloudCover"]        = df["cloudCover"] / 100.0
    if "precipProbability" in df.columns:
        df["precipProbability"] = df["precipProbability"] / 100.0
    if "windSpeed"         in df.columns:
        df["windSpeed"]         = df["windSpeed"] * 0.621371   # km/h → mph
    if "visibility"        in df.columns:
        df["visibility"]        = (df["visibility"] / 1609.34).clip(upper=10.0)
    return df


def _parse_response(data: dict) -> pd.DataFrame:
    """Parses an Open-Meteo hourly JSON response into a clean DataFrame."""
    hourly = data["hourly"]
    df     = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    df.index.name = "time"

    rename = {k: v for k, v in VARIABLE_MAP.items() if k in df.columns}
    df = df.rename(columns=rename)

    # precipProbability not available in historical archive — default to 0
    if "precipProbability" not in df.columns:
        df["precipProbability"] = 0.0

    df = _normalise_units(df)
    df = df.ffill().bfill().fillna(0)
    return df


def fetch_historical_weather(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Fetches hourly historical weather for a location and date range.

    Args:
        lat, lon:   Coordinates from geocode()
        start_date: 'YYYY-MM-DD'
        end_date:   'YYYY-MM-DD'

    Returns:
        DataFrame with DatetimeIndex and weather feature columns.
    """
    url    = "https://archive.open-meteo.com/v1/archive"
    params = {
        "latitude":        lat,
        "longitude":       lon,
        "start_date":      start_date,
        "end_date":        end_date,
        "hourly":          ",".join(HISTORICAL_VARS),
        "wind_speed_unit": "kmh",
        "timezone":        "UTC",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return _parse_response(resp.json())


def fetch_forecast_weather(
    lat: float,
    lon: float,
    forecast_date: str,
) -> pd.DataFrame:
    """
    Fetches hourly weather forecast for a single day (24 rows).

    Args:
        lat, lon:       Coordinates from geocode()
        forecast_date:  'YYYY-MM-DD'

    Returns:
        DataFrame with 24 rows of weather features for that day.
    """
    url    = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":        lat,
        "longitude":       lon,
        "hourly":          ",".join(FORECAST_VARS),
        "wind_speed_unit": "kmh",
        "timezone":        "UTC",
        "forecast_days":   16,
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()

    df     = _parse_response(resp.json())
    target = pd.Timestamp(forecast_date)
    df     = df[(df.index >= target) & (df.index < target + pd.Timedelta(days=1))]
    return df


def merge_weather_with_meter(
    meter_df: pd.DataFrame,
    weather_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-joins weather columns onto the meter DataFrame by timestamp index.
    Missing weather rows are forward-filled then zero-filled.

    Args:
        meter_df:   Hourly meter data with 'use [kW]' column and DatetimeIndex
        weather_df: Weather DataFrame from fetch_historical_weather()

    Returns:
        Merged DataFrame with weather columns added.
    """
    if weather_df is None or len(weather_df) == 0:
        return meter_df

    cols   = [c for c in WEATHER_COLS if c in weather_df.columns]
    merged = meter_df.join(weather_df[cols], how="left")
    merged[cols] = merged[cols].ffill().bfill().fillna(0)
    return merged
