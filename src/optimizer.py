"""
optimizer.py

Translates a 24-hour consumption forecast into plain English
appliance scheduling recommendations.

Pure rule-based logic that scans the forecast curve,
identifies optimal windows, and maps them to specific appliances.
The user never sees a number. They see a decision.
"""

import pandas as pd
import numpy as np
from typing import List, Dict


# Appliance Definitions

APPLIANCES: Dict[str, Dict] = {
    "Dishwasher": {
        "run_hours": 1.5,
        "wattage": 1200,
        "flexible": True,
        "description": "Can run any time — dishes don't care what hour it is"
    },
    "Washing Machine": {
        "run_hours": 1.0,
        "wattage": 2000,
        "flexible": True,
        "description": "Full cycle typically 45-60 minutes"
    },
    "Tumble Dryer": {
        "run_hours": 1.0,
        "wattage": 2500,
        "flexible": True,
        "description": "High draw — most benefit from off-peak scheduling"
    },
    "EV Charger": {
        "run_hours": 4.0,
        "wattage": 7200,
        "flexible": True,
        "description": "Overnight charging — widest scheduling flexibility"
    },
    "Microwave": {
        "run_hours": 0.25,
        "wattage": 1100,
        "flexible": False,
        "description": "Short duration — timing impact minimal"
    },
}

# Time-of-use tariff assumptions (pence or cents per kWh)
PEAK_RATE     = 28.0   # peak hours: 7am to 9pm
OFF_PEAK_RATE = 12.0   # off-peak: 9pm to 7am


# Core Optimizer Logic

def find_optimal_window(
    forecast: pd.DataFrame,
    run_hours: float,
    exclude_hours: list = None
) -> int:
    """
    Finds the starting hour with the lowest average predicted consumption
    over a window of run_hours length.

    Optionally excludes already-assigned hours so multiple appliances
    receive distinct, staggered recommendations.

    Args:
        forecast:      DataFrame with 'yhat' column indexed by timestamps.
        run_hours:     Duration of the appliance cycle in hours.
        exclude_hours: List of hours already assigned to other appliances.

    Returns:
        Integer hour (0-23) representing the best available start time.
    """
    if exclude_hours is None:
        exclude_hours = []

    window = max(1, int(np.ceil(run_hours)))
    yhat   = forecast['yhat'].values
    n      = len(yhat)

    best_hour = 0
    best_avg  = float('inf')

    for start in range(n - window + 1):
        # Skip windows that overlap with already-assigned hours
        window_hours = list(range(start, start + window))
        if any(h in exclude_hours for h in window_hours):
            continue

        avg = np.mean(yhat[start:start + window])
        if avg < best_avg:
            best_avg  = avg
            best_hour = start

    return best_hour


def find_peak_hour(forecast: pd.DataFrame) -> int:
    """
    Returns the hour with the highest predicted consumption.

    Args:
        forecast: DataFrame with 'yhat' column indexed by hour timestamps.

    Returns:
        Integer hour (0-23) of predicted peak consumption.
    """
    return int(forecast['yhat'].idxmax().hour)


def find_trough_hour(forecast: pd.DataFrame) -> int:
    """
    Returns the hour with the lowest predicted consumption.

    Args:
        forecast: DataFrame with 'yhat' column indexed by hour timestamps.

    Returns:
        Integer hour (0-23) of predicted lowest consumption.
    """
    return int(forecast['yhat'].idxmin().hour)


def estimate_savings(
    appliance_name: str,
    optimal_hour: int,
    peak_hour: int,
    forecast: pd.DataFrame = None
) -> float:
    """
    Estimates cost saving in pence/cents from running an appliance
    at the optimal hour vs the peak hour.

    Uses forecast-relative pricing — the optimal window is compared
    against the peak hour consumption level, not just time-of-day tariff.
    This rewards any low-demand window, not just overnight hours.

    Args:
        appliance_name: Name matching a key in APPLIANCES dict.
        optimal_hour:   Recommended start hour (0-23).
        peak_hour:      Peak consumption hour to compare against (0-23).
        forecast:       Optional forecast DataFrame for relative pricing.

    Returns:
        Estimated saving as a float. Positive means the optimal window
        is cheaper than the peak hour.
    """
    appliance  = APPLIANCES[appliance_name]
    wattage    = appliance['wattage']
    run_hours  = appliance['run_hours']
    energy_kwh = (wattage / 1000) * run_hours

    if forecast is not None:
        # Relative pricing based on forecast consumption level
        # Higher predicted consumption = higher effective rate
        yhat        = forecast['yhat'].values
        peak_val    = yhat.max()
        trough_val  = yhat.min()
        value_range = max(peak_val - trough_val, 0.01)

        optimal_consumption = yhat[optimal_hour]
        peak_consumption    = yhat[peak_hour]

        # Scale rate between OFF_PEAK_RATE and PEAK_RATE based on consumption
        optimal_rate = OFF_PEAK_RATE + (
            (optimal_consumption - trough_val) / value_range
        ) * (PEAK_RATE - OFF_PEAK_RATE)

        peak_rate = OFF_PEAK_RATE + (
            (peak_consumption - trough_val) / value_range
        ) * (PEAK_RATE - OFF_PEAK_RATE)

    else:
        optimal_rate = OFF_PEAK_RATE if (
            optimal_hour >= 21 or optimal_hour < 7
        ) else PEAK_RATE
        peak_rate = OFF_PEAK_RATE if (
            peak_hour >= 21 or peak_hour < 7
        ) else PEAK_RATE

    optimal_cost = energy_kwh * optimal_rate / 100
    peak_cost    = energy_kwh * peak_rate    / 100

    return round(peak_cost - optimal_cost, 2)


def format_hour(hour: int) -> str:
    """
    Converts a 24-hour integer to a readable 12-hour time string.

    Args:
        hour: Integer 0-23.

    Returns:
        String like '11:00 AM' or '9:00 PM'.
    """
    if hour == 0:
        return "12:00 AM"
    elif hour < 12:
        return f"{hour}:00 AM"
    elif hour == 12:
        return "12:00 PM"
    else:
        return f"{hour - 12}:00 PM"


# Recommendation Generator

def generate_recommendations(
    forecast: pd.DataFrame,
    selected_appliances: List[str]
) -> List[Dict]:
    """
    Generates plain English scheduling recommendations for each
    selected appliance based on the 24-hour forecast.

    Args:
        forecast:             DataFrame with 'yhat' column from make_forecast().
        selected_appliances:  List of appliance names from APPLIANCES keys.

    Returns:
        List of recommendation dicts, one per appliance. Each dict contains:
            - appliance:     Appliance name
            - recommended_time: Human-readable time string
            - recommended_hour: Integer hour (0-23)
            - peak_time:     Human-readable peak time string
            - saving:        Estimated cost saving float
            - message:       Plain English recommendation string
            - avoid_message: Plain English warning string
    """
    if not selected_appliances:
        return []

    peak_hour   = find_peak_hour(forecast)
    trough_hour = find_trough_hour(forecast)
    peak_time   = format_hour(peak_hour)
    trough_time = format_hour(trough_hour)

    peak_value   = forecast['yhat'].max()
    trough_value = forecast['yhat'].min()

    recommendations = []

    assigned_hours = []

    for appliance_name in selected_appliances:
        if appliance_name not in APPLIANCES:
            continue

        appliance = APPLIANCES[appliance_name]

        # Find best window excluding already-assigned hours
        optimal_hour = find_optimal_window(
            forecast,
            appliance['run_hours'],
            exclude_hours=assigned_hours
        )
        optimal_time = format_hour(optimal_hour)

        # Mark these hours as assigned so next appliance gets a different window
        window_size = max(1, int(np.ceil(appliance['run_hours'])))
        assigned_hours.extend(range(optimal_hour, optimal_hour + window_size))

        # Estimate savings using forecast-relative pricing
        saving = estimate_savings(
            appliance_name, optimal_hour, peak_hour, forecast
        )

        # Calculate how much lower the optimal window is vs peak
        optimal_consumption = forecast['yhat'].iloc[optimal_hour]
        pct_below_peak = ((peak_value - optimal_consumption) / peak_value) * 100

        # Build the plain English message
        if saving > 0:
            saving_str = f"Saves approximately ${saving:.2f} vs running at {peak_time}."
        else:
            saving_str = f"Similar cost to peak — but reduces grid stress."

        message = (
            f"Run your {appliance_name} at {optimal_time}. "
            f"That is the lowest demand window — "
            f"{pct_below_peak:.0f}% below the peak at {peak_time}. "
            f"{saving_str}"
        )

        avoid_message = (
            f"Avoid running your {appliance_name} between "
            f"{format_hour(max(0, peak_hour - 1))} and "
            f"{format_hour(min(23, peak_hour + 1))}. "
            f"That is when your home uses the most energy."
        )

        recommendations.append({
            'appliance':        appliance_name,
            'recommended_time': optimal_time,
            'recommended_hour': optimal_hour,
            'peak_time':        peak_time,
            'peak_hour':        peak_hour,
            'trough_time':      trough_time,
            'trough_hour':      trough_hour,
            'saving':           saving,
            'message':          message,
            'avoid_message':    avoid_message,
        })

    return recommendations


def get_daily_summary(forecast: pd.DataFrame) -> Dict:
    """
    Produces a high-level summary of the day's forecast for the
    dashboard summary panel.

    Args:
        forecast: DataFrame with 'yhat' column from make_forecast().

    Returns:
        Dictionary with peak, trough, and total consumption stats.
    """
    peak_hour   = find_peak_hour(forecast)
    trough_hour = find_trough_hour(forecast)

    return {
        'peak_hour':        peak_hour,
        'peak_time':        format_hour(peak_hour),
        'peak_value':       round(float(forecast['yhat'].max()), 3),
        'trough_hour':      trough_hour,
        'trough_time':      format_hour(trough_hour),
        'trough_value':     round(float(forecast['yhat'].min()), 3),
        'total_kwh':        round(float(forecast['yhat'].sum()), 2),
        'mean_kwh':         round(float(forecast['yhat'].mean()), 3),
    }