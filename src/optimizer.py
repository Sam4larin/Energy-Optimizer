"""
src/optimizer.py

Translates a 24-hour consumption forecast into appliance scheduling
recommendations and budget planning guidance.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


APPLIANCE_LIBRARY: Dict[str, Dict] = {
    "Dishwasher": {"wattage": 1200, "run_hours": 1.5, "flexible": True, "category": "Kitchen"},
    "Microwave": {"wattage": 1100, "run_hours": 0.1, "flexible": False, "category": "Kitchen"},
    "Oven": {"wattage": 2000, "run_hours": 1.0, "flexible": False, "category": "Kitchen"},
    "Kettle": {"wattage": 2000, "run_hours": 0.05, "flexible": False, "category": "Kitchen"},
    "Fridge Freezer": {"wattage": 150, "run_hours": 24.0, "flexible": False, "category": "Kitchen"},
    "Washing Machine": {"wattage": 2000, "run_hours": 1.0, "flexible": True, "category": "Laundry"},
    "Tumble Dryer": {"wattage": 2500, "run_hours": 1.0, "flexible": True, "category": "Laundry"},
    "Washer-Dryer": {"wattage": 2200, "run_hours": 2.0, "flexible": True, "category": "Laundry"},
    "Iron": {"wattage": 2000, "run_hours": 0.5, "flexible": True, "category": "Laundry"},
    "Electric Shower": {"wattage": 8500, "run_hours": 0.15, "flexible": True, "category": "Heating & Cooling"},
    "Immersion Heater": {"wattage": 3000, "run_hours": 2.0, "flexible": True, "category": "Heating & Cooling"},
    "Heat Pump": {"wattage": 1500, "run_hours": 8.0, "flexible": True, "category": "Heating & Cooling"},
    "Air Conditioning": {"wattage": 1500, "run_hours": 4.0, "flexible": True, "category": "Heating & Cooling"},
    "Electric Heater": {"wattage": 2000, "run_hours": 4.0, "flexible": True, "category": "Heating & Cooling"},
    "EV Charger (7kW)": {"wattage": 7200, "run_hours": 4.0, "flexible": True, "category": "Transport"},
    "EV Charger (3kW)": {"wattage": 3000, "run_hours": 8.0, "flexible": True, "category": "Transport"},
    "TV": {"wattage": 100, "run_hours": 4.0, "flexible": False, "category": "Entertainment"},
    "Gaming Console": {"wattage": 200, "run_hours": 2.0, "flexible": True, "category": "Entertainment"},
    "Desktop Computer": {"wattage": 300, "run_hours": 4.0, "flexible": True, "category": "Entertainment"},
    "Pool Pump": {"wattage": 750, "run_hours": 6.0, "flexible": True, "category": "Other"},
    "Hot Tub": {"wattage": 3000, "run_hours": 2.0, "flexible": True, "category": "Other"},
}

# Backward compatibility for tests and older callers.
APPLIANCES = APPLIANCE_LIBRARY

DEFAULT_PEAK_RATE = 28.0
DEFAULT_OFF_PEAK_RATE = 12.0
DEFAULT_CURRENCY = "£"


def get_appliance_categories() -> Dict[str, List[str]]:
    categories: Dict[str, List[str]] = {}
    for name, info in APPLIANCE_LIBRARY.items():
        categories.setdefault(info['category'], []).append(name)
    return categories


def get_flexible_appliances(
    selected: List[str],
    custom: Optional[List[Dict]] = None,
) -> List[Dict]:
    result: List[Dict] = []
    for name in selected:
        if name in APPLIANCE_LIBRARY and APPLIANCE_LIBRARY[name]['flexible']:
            info = APPLIANCE_LIBRARY[name]
            result.append({
                'name': name,
                'wattage': info['wattage'],
                'run_hours': info['run_hours'],
            })
    if custom:
        for item in custom:
            if item.get('name') and item.get('wattage') and item.get('run_hours'):
                result.append({
                    'name': item['name'],
                    'wattage': float(item['wattage']),
                    'run_hours': float(item['run_hours']),
                })
    return result


def _coerce_appliances(appliances: List) -> List[Dict]:
    coerced: List[Dict] = []
    for appliance in appliances:
        if isinstance(appliance, str) and appliance in APPLIANCE_LIBRARY:
            info = APPLIANCE_LIBRARY[appliance]
            coerced.append({
                'name': appliance,
                'wattage': info['wattage'],
                'run_hours': info['run_hours'],
            })
        elif isinstance(appliance, dict):
            name = appliance.get('name') or appliance.get('appliance')
            wattage = appliance.get('wattage')
            run_hours = appliance.get('run_hours')
            if name and wattage and run_hours:
                coerced.append({
                    'name': name,
                    'wattage': float(wattage),
                    'run_hours': float(run_hours),
                })
    return coerced


def find_optimal_window(
    forecast: pd.DataFrame,
    run_hours: float,
    exclude_hours: Optional[List[int]] = None,
) -> int:
    if exclude_hours is None:
        exclude_hours = []

    window = max(1, int(np.ceil(run_hours)))
    yhat = forecast['yhat'].values
    n = len(yhat)
    best_hour: Optional[int] = None
    best_avg = float('inf')

    for start in range(n - window + 1):
        window_hours = list(range(start, start + window))
        if any(h in exclude_hours for h in window_hours):
            continue
        avg = float(np.mean(yhat[start:start + window]))
        if avg < best_avg:
            best_avg = avg
            best_hour = start

    if best_hour is not None:
        return best_hour

    # Fall back to the best available window if the exclusion list blocks all slots.
    for start in range(n - window + 1):
        avg = float(np.mean(yhat[start:start + window]))
        if avg < best_avg:
            best_avg = avg
            best_hour = start

    return 0 if best_hour is None else best_hour


def find_peak_hour(forecast: pd.DataFrame) -> int:
    return int(forecast['yhat'].idxmax().hour)


def find_trough_hour(forecast: pd.DataFrame) -> int:
    return int(forecast['yhat'].idxmin().hour)


def format_hour(hour: int) -> str:
    if hour == 0:
        return "12:00 AM"
    if hour < 12:
        return f"{hour}:00 AM"
    if hour == 12:
        return "12:00 PM"
    return f"{hour - 12}:00 PM"


def estimate_savings(
    wattage: float,
    run_hours: float,
    optimal_hour: int,
    peak_hour: int,
    forecast: pd.DataFrame,
    rate_per_kwh: float,
) -> Dict:
    energy_kwh = (wattage / 1000) * run_hours
    yhat = forecast['yhat'].values

    peak_load = float(yhat.max())
    trough_load = float(yhat.min())
    load_range = max(peak_load - trough_load, 0.001)

    def effective_rate(hour: int) -> float:
        load_ratio = (yhat[hour] - trough_load) / load_range
        return rate_per_kwh * (1.0 + 0.5 * load_ratio)

    cost_at_optimal = energy_kwh * effective_rate(optimal_hour)
    cost_at_peak = energy_kwh * effective_rate(peak_hour)
    saving = max(cost_at_peak - cost_at_optimal, 0.0)
    pct_saving = (saving / cost_at_peak * 100) if cost_at_peak > 0 else 0.0
    pct_below_peak = (
        (yhat[peak_hour] - yhat[optimal_hour]) / yhat[peak_hour] * 100
        if yhat[peak_hour] > 0 else 0.0
    )

    return {
        'energy_kwh': round(energy_kwh, 3),
        'cost_at_optimal': round(cost_at_optimal, 2),
        'cost_at_peak': round(cost_at_peak, 2),
        'saving': round(saving, 2),
        'pct_saving': round(pct_saving, 1),
        'pct_below_peak': round(pct_below_peak, 1),
    }


def generate_recommendations(
    forecast: pd.DataFrame,
    appliances: List,
    peak_rate: float = DEFAULT_PEAK_RATE,
    off_peak_rate: float = DEFAULT_OFF_PEAK_RATE,
    currency: str = DEFAULT_CURRENCY,
    show_cost_savings: bool = True,
) -> List[Dict]:
    if forecast is None or len(forecast) == 0:
        return []

    normalized_appliances = _coerce_appliances(appliances)
    if not normalized_appliances:
        return []

    peak_hour = find_peak_hour(forecast)
    trough_hour = find_trough_hour(forecast)
    peak_time = format_hour(peak_hour)
    trough_time = format_hour(trough_hour)
    base_rate = (peak_rate + off_peak_rate) / 2

    recommendations: List[Dict] = []
    assigned_hours: List[int] = []

    for appliance in normalized_appliances:
        name = appliance['name']
        wattage = appliance['wattage']
        run_hours = appliance['run_hours']

        optimal_hour = find_optimal_window(forecast, run_hours, exclude_hours=assigned_hours)
        optimal_time = format_hour(optimal_hour)

        window_size = max(1, int(np.ceil(run_hours)))
        assigned_hours.extend(range(optimal_hour, optimal_hour + window_size))

        savings = estimate_savings(
            wattage, run_hours, optimal_hour, peak_hour, forecast, base_rate
        )

        if show_cost_savings and savings['saving'] > 0:
            saving_str = (
                f"Saves {currency}{savings['saving']:.2f} "
                f"({savings['pct_saving']:.0f}% cheaper) vs running at {peak_time}."
            )
        elif show_cost_savings:
            saving_str = "Similar cost at any hour, but it still reduces grid stress."
        else:
            saving_str = "This mainly reduces demand rather than cost on a flat tariff."

        message = (
            f"Run your {name} at {optimal_time}. "
            f"Demand is {savings['pct_below_peak']:.0f}% lower than the {peak_time} peak. "
            f"{saving_str}"
        )

        if show_cost_savings:
            avoid_message = (
                f"Avoid {format_hour(max(0, peak_hour - 1))} to "
                f"{format_hour(min(23, peak_hour + 2))}. "
                f"That window costs {currency}{savings['cost_at_peak']:.2f} "
                f"vs {currency}{savings['cost_at_optimal']:.2f} at the best time."
            )
        else:
            avoid_message = (
                f"Avoid {format_hour(max(0, peak_hour - 1))} to "
                f"{format_hour(min(23, peak_hour + 2))}. "
                f"That window is the highest-demand period of the day."
            )

        recommendations.append({
            'appliance': name,
            'recommended_time': optimal_time,
            'recommended_hour': optimal_hour,
            'peak_time': peak_time,
            'peak_hour': peak_hour,
            'trough_time': trough_time,
            'trough_hour': trough_hour,
            'saving': savings['saving'],
            'pct_saving': savings['pct_saving'],
            'cost_at_optimal': savings['cost_at_optimal'],
            'cost_at_peak': savings['cost_at_peak'],
            'energy_kwh': savings['energy_kwh'],
            'currency': currency,
            'message': message,
            'avoid_message': avoid_message,
        })

    return recommendations


def get_daily_summary(forecast: pd.DataFrame) -> Dict:
    peak_hour = find_peak_hour(forecast)
    trough_hour = find_trough_hour(forecast)
    return {
        'peak_hour': peak_hour,
        'peak_time': format_hour(peak_hour),
        'peak_value': round(float(forecast['yhat'].max()), 3),
        'trough_hour': trough_hour,
        'trough_time': format_hour(trough_hour),
        'trough_value': round(float(forecast['yhat'].min()), 3),
        'total_kwh': round(float(forecast['yhat'].sum()), 2),
        'mean_kwh': round(float(forecast['yhat'].mean()), 3),
    }


def _build_appliance_impact_list(
    user_appliances: Optional[List[Dict]],
) -> List[Dict]:
    """
    Builds the appliance list used for the budget impact table and cuts suggestions.

    Priority order:
    1. User-selected appliances from the sidebar picker (with their exact wattage
       and run_hours — these are the real values for their home).
    2. Any appliance in APPLIANCE_LIBRARY that is NOT already covered by the user's
       selection, added as a fallback so the table always has enough rows to be useful.

    When the user has selected appliances, the table is personalised to THEIR home.
    When they haven't selected any (e.g. first visit, demo mode), it falls back to
    a representative set drawn from APPLIANCE_LIBRARY defaults.

    Args:
        user_appliances: List of dicts from get_flexible_appliances(), each with
                         'name', 'wattage', 'run_hours'. May be None or empty.

    Returns:
        List of dicts with keys: name, wattage, daily_hours, flexible.
    """
    # Fallback set — covers both flexible and non-flexible for a realistic table.
    # daily_hours here means typical daily usage (not per-cycle run_hours).
    FALLBACK_DAILY_HOURS: Dict[str, float] = {
        "Electric Shower":   0.30,   # ~18 min/day
        "Washing Machine":   0.50,   # ~1 cycle every 2 days
        "Tumble Dryer":      0.50,
        "Oven":              1.00,
        "Air Conditioning":  4.00,
        "Electric Heater":   4.00,
        "TV":                4.00,
        "Fridge Freezer":   24.00,   # always on
        "Desktop Computer":  4.00,
        "Immersion Heater":  1.00,
        "EV Charger (7kW)":  2.00,   # typical overnight partial charge
        "EV Charger (3kW)":  4.00,
        "Heat Pump":         8.00,
        "Dishwasher":        0.75,   # once per day ~1.5h but not every day
    }

    result: List[Dict] = []
    covered_names: set = set()

    # Step 1: add user-selected appliances using their actual wattage and run_hours.
    # run_hours in APPLIANCE_LIBRARY is per-cycle. We treat it as daily_hours
    # (the user selected it because they run it roughly that often per day).
    if user_appliances:
        for appl in user_appliances:
            name = appl['name']
            wattage = float(appl['wattage'])
            run_hours = float(appl['run_hours'])

            # Look up flexibility from library; custom appliances default to flexible.
            lib_entry = APPLIANCE_LIBRARY.get(name, {})
            flexible = lib_entry.get('flexible', True)

            result.append({
                'name': name,
                'wattage': wattage,
                'daily_hours': run_hours,
                'flexible': flexible,
                'user_selected': True,
            })
            covered_names.add(name)

    # Step 2: fill in with library fallbacks that the user did NOT select,
    # so the table always has enough variety to be informative.
    # Only add if we have fewer than 8 entries (keeps the table readable).
    if len(result) < 8:
        for name, daily_hours in FALLBACK_DAILY_HOURS.items():
            if name in covered_names:
                continue
            lib_entry = APPLIANCE_LIBRARY.get(name, {})
            if not lib_entry:
                continue
            result.append({
                'name': name,
                'wattage': float(lib_entry['wattage']),
                'daily_hours': daily_hours,
                'flexible': lib_entry.get('flexible', True),
                'user_selected': False,
            })
            if len(result) >= 10:
                break

    return result


def calculate_budget_plan(
    historical_df: pd.DataFrame,
    kwh_budget: float,
    target_days: int,
    rate_per_kwh: float,
    currency: str = DEFAULT_CURRENCY,
    forecast: Optional[pd.DataFrame] = None,
    user_appliances: Optional[List[Dict]] = None,
) -> Dict:
    """
    Produces a personalised budget analysis.

    Args:
        historical_df:   Hourly meter DataFrame with 'use [kW]' column.
        kwh_budget:      Total kWh budget over the target period.
        target_days:     Number of days the budget must cover.
        rate_per_kwh:    Electricity rate (pence/cents per kWh).
        currency:        Currency symbol string.
        forecast:        Optional 24-hour forecast DataFrame (unused directly
                         but kept for API compatibility).
        user_appliances: List of appliance dicts from get_flexible_appliances().
                         When provided, the impact table and cuts are based on
                         the user's actual appliances rather than generic defaults.
                         Each dict must have 'name', 'wattage', 'run_hours'.
    """
    avg_hourly_kwh = float(historical_df['use [kW]'].mean())
    avg_daily_kwh = avg_hourly_kwh * 24
    avg_daily_cost = avg_daily_kwh * rate_per_kwh

    budget_daily_kwh = kwh_budget / target_days
    budget_daily_cost = budget_daily_kwh * rate_per_kwh
    total_cost = kwh_budget * rate_per_kwh
    days_at_current = kwh_budget / avg_daily_kwh if avg_daily_kwh > 0 else 0.0

    daily_kwh_gap = avg_daily_kwh - budget_daily_kwh
    daily_cost_gap = daily_kwh_gap * rate_per_kwh
    pct_reduction = (daily_kwh_gap / avg_daily_kwh * 100) if avg_daily_kwh > 0 else 0.0

    # Most expensive hours from actual meter data (always personalised)
    hourly_avg = historical_df.groupby(historical_df.index.hour)['use [kW]'].mean()
    hourly_cost = hourly_avg * rate_per_kwh
    top_expensive = hourly_cost.nlargest(3)
    expensive_hours = [
        {
            'hour': int(hour),
            'time': format_hour(int(hour)),
            'kwh': round(float(hourly_avg[hour]), 3),
            'cost': round(float(hourly_cost[hour]), 3),
            'currency': currency,
        }
        for hour in top_expensive.index
    ]

    # Build appliance list — personalised if user selected appliances, else defaults
    raw_appliances = _build_appliance_impact_list(user_appliances)

    appliance_impacts: List[Dict] = []
    for appl in raw_appliances:
        daily_kwh = (appl['wattage'] / 1000) * appl['daily_hours']
        daily_cost = daily_kwh * rate_per_kwh
        pct_of_budget = (daily_kwh / budget_daily_kwh * 100) if budget_daily_kwh > 0 else 0.0
        appliance_impacts.append({
            'name': appl['name'],
            'daily_kwh': round(daily_kwh, 2),
            'daily_cost': round(daily_cost, 2),
            'pct_of_budget': round(pct_of_budget, 1),
            'flexible': appl['flexible'],
            'user_selected': appl.get('user_selected', False),
            'currency': currency,
        })

    appliance_impacts.sort(key=lambda item: item['daily_cost'], reverse=True)

    # Cuts: only suggest flexible appliances, prioritised by highest impact
    cuts: List[Dict] = []
    cumulative_saving = 0.0
    for appl in appliance_impacts:
        if not appl['flexible']:
            continue
        if daily_kwh_gap > 0 and cumulative_saving >= daily_kwh_gap:
            break
        if appl['daily_kwh'] >= 1.0:
            reduced_daily = appl['daily_kwh'] * (3 / 7)
            saving = appl['daily_kwh'] - reduced_daily
            action = "Use 3× per week instead of daily"
        else:
            saving = appl['daily_kwh'] * 0.5
            action = "Reduce usage by half"
        saving_cost = saving * rate_per_kwh
        cuts.append({
            'appliance': appl['name'],
            'action': action,
            'daily_saving_kwh': round(saving, 3),
            'daily_saving_cost': round(saving_cost, 2),
            'currency': currency,
        })
        cumulative_saving += saving

    achievable = daily_kwh_gap <= 0 or pct_reduction <= 35
    stretch_target = 35 < pct_reduction <= 60

    if daily_kwh_gap <= 0:
        verdict = "Your budget is comfortable. At current usage it will last longer than planned."
    elif achievable:
        verdict = f"Achievable. You need to cut {pct_reduction:.0f}%."
    elif stretch_target:
        verdict = f"Stretch target. Cutting {pct_reduction:.0f}% is possible but demanding."
    else:
        verdict = (
            f"Very difficult. Cutting {pct_reduction:.0f}% would mean major lifestyle changes."
        )

    # Flag whether the table is personalised or generic
    is_personalised = bool(user_appliances)

    return {
        'kwh_budget': round(kwh_budget, 2),
        'target_days': target_days,
        'total_cost': round(total_cost, 2),
        'budget_daily_kwh': round(budget_daily_kwh, 3),
        'budget_daily_cost': round(budget_daily_cost, 2),
        'avg_daily_kwh': round(avg_daily_kwh, 3),
        'avg_daily_cost': round(avg_daily_cost, 2),
        'daily_kwh_gap': round(daily_kwh_gap, 3),
        'daily_cost_gap': round(daily_cost_gap, 2),
        'pct_reduction': round(pct_reduction, 1),
        'days_at_current': round(days_at_current, 1),
        'verdict': verdict,
        'achievable': achievable,
        'stretch': stretch_target,
        'expensive_hours': expensive_hours,
        'appliance_impacts': appliance_impacts[:8],
        'cuts': cuts,
        'currency': currency,
        'rate_per_kwh': rate_per_kwh,
        'is_personalised': is_personalised,
    }