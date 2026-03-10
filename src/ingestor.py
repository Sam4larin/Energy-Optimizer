"""
src/ingestor.py

CSV ingestion and normalisation for user-uploaded smart meter data.

Handles arbitrary CSV formats by auto-detecting timestamp and consumption
columns, validating data quality, and normalising to the standard hourly
format the rest of the pipeline expects.
"""

from typing import Dict, List, Optional, Tuple

import pandas as pd


TIMESTAMP_PATTERNS = [
    'time', 'date', 'timestamp', 'datetime', 'period',
    'interval', 'reading_date', 'settlementdate', 'start',
    'end', 'hour', 'day', 'recorded', 'at', 'when'
]

CONSUMPTION_PATTERNS = [
    'kwh', 'consumption', 'usage', 'energy', 'use',
    'import', 'reading', 'watt', 'power', 'demand',
    'electricity', 'total', 'value', 'amount', 'units',
    'actual', 'meter', 'active', 'anytime', 'offpeak',
    'global', 'submetering',
]

WEATHER_FEATURE_COLS_CHECK = [
    'temperature', 'humidity', 'windSpeed', 'cloudCover',
    'precipIntensity', 'dewPoint', 'pressure', 'visibility',
    'apparentTemperature', 'windBearing', 'precipProbability',
]

WEATHER_COLS_FAHRENHEIT = ['temperature', 'apparentTemperature', 'dewPoint']


def detect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Attempts to auto-detect timestamp and consumption columns.
    """
    cols_lower = {
        col: col.lower().replace(' ', '_').replace('-', '_')
        for col in df.columns
    }

    timestamp_scores: Dict[str, int] = {}
    consumption_scores: Dict[str, int] = {}

    for original, lower in cols_lower.items():
        ts_score = sum(p in lower for p in TIMESTAMP_PATTERNS)
        con_score = sum(p in lower for p in CONSUMPTION_PATTERNS)

        if ts_score > 0:
            try:
                sample_vals = df[original].dropna().iloc[:5]
                if pd.api.types.is_numeric_dtype(df[original]):
                    sample_val = float(sample_vals.iloc[0]) if len(sample_vals) > 0 else 0
                    if sample_val > 1_000_000_000:
                        parsed = pd.to_datetime(sample_vals, unit='s', errors='coerce')
                        if parsed.notna().all() and parsed.dt.year.between(1990, 2100).all():
                            ts_score += 3
                    else:
                        pd.to_datetime(sample_vals)
                        ts_score += 3
                else:
                    pd.to_datetime(sample_vals)
                    ts_score += 3
            except Exception:
                pass

        if con_score > 0 and pd.api.types.is_numeric_dtype(df[original]):
            sample = df[original].dropna()
            if len(sample) > 0 and sample.mean() < 100:
                con_score += 2

        if ts_score > 0:
            timestamp_scores[original] = ts_score
        if con_score > 0:
            consumption_scores[original] = con_score

    if not timestamp_scores:
        for col in df.columns:
            try:
                pd.to_datetime(df[col].dropna().iloc[:5])
                timestamp_scores[col] = 1
                break
            except Exception:
                pass

    def is_confident(scores: Dict[str, int]) -> bool:
        if len(scores) < 2:
            return bool(scores)
        sorted_vals = sorted(scores.values(), reverse=True)
        return sorted_vals[0] >= 2 and sorted_vals[0] > sorted_vals[1]

    ts_col = max(timestamp_scores, key=timestamp_scores.get) if timestamp_scores else None
    con_col = max(consumption_scores, key=consumption_scores.get) if consumption_scores else None

    return {
        'timestamp_col': ts_col,
        'consumption_col': con_col,
        'timestamp_confident': is_confident(timestamp_scores),
        'consumption_confident': is_confident(consumption_scores),
    }


def get_all_columns(df: pd.DataFrame) -> List[str]:
    return list(df.columns)


def _parse_timestamp_column(series: pd.Series) -> Tuple[pd.Series, List[str]]:
    warnings: List[str] = []
    sample = series.dropna().iloc[:10] if len(series.dropna()) >= 10 else series.dropna()

    if pd.api.types.is_numeric_dtype(series):
        sample_val = float(sample.iloc[0]) if len(sample) > 0 else 0
        if sample_val > 1_000_000_000_000:
            parsed = pd.to_datetime(series, unit='ms', errors='coerce')
            if parsed.notna().sum() > len(series) * 0.9:
                warnings.append("Unix millisecond timestamps detected and converted to datetime.")
                return parsed, warnings
        if sample_val > 1_000_000_000:
            parsed = pd.to_datetime(series, unit='s', errors='coerce')
            if parsed.notna().sum() > len(series) * 0.9:
                warnings.append("Unix epoch timestamps detected and converted to datetime.")
                return parsed, warnings

    parsed = pd.to_datetime(series, errors='coerce')
    valid = parsed.dropna()
    if len(valid) > 0:
        year_min = valid.dt.year.min()
        year_max = valid.dt.year.max()
        if year_min < 1990 or year_max > 2100:
            retry = pd.to_datetime(series, unit='s', errors='coerce')
            retry_valid = retry.dropna()
            if len(retry_valid) > 0:
                retry_year_min = retry_valid.dt.year.min()
                retry_year_max = retry_valid.dt.year.max()
                if 1990 <= retry_year_min and retry_year_max <= 2100:
                    warnings.append("Unix epoch timestamps detected and converted to datetime.")
                    return retry, warnings

    return parsed, warnings


def read_csv_smart(file_obj) -> Tuple[pd.DataFrame, List[str]]:
    warnings: List[str] = []

    try:
        if hasattr(file_obj, 'read'):
            raw = file_obj.read(2048)
            if isinstance(raw, bytes):
                raw = raw.decode('utf-8', errors='replace')
            file_obj.seek(0)
        else:
            raw = ""
    except Exception:
        raw = ""

    sep = ','
    if raw:
        first_line = raw.split('\n')[0] if '\n' in raw else raw
        semicolons = first_line.count(';')
        tabs = first_line.count('\t')
        commas = first_line.count(',')
        if semicolons > commas and semicolons >= tabs:
            sep = ';'
            warnings.append("Semicolon-separated file detected and parsed correctly.")
        elif tabs > commas:
            sep = '\t'
            warnings.append("Tab-separated file detected and parsed correctly.")

    try:
        df = pd.read_csv(
            file_obj,
            sep=sep,
            na_values=['?', 'NA', 'N/A', 'nan', 'NaN', '', 'null', 'NULL'],
            low_memory=False,
        )
    except Exception as exc:
        raise ValueError(f"Could not read file: {exc}") from exc

    return df, warnings


def normalise_to_hourly(
    df: pd.DataFrame,
    timestamp_col: str,
    consumption_col: str,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Normalises raw meter data to a standard hourly DataFrame.
    """
    warnings: List[str] = []
    df = df.copy()

    col_lower = timestamp_col.lower().strip()
    time_col_candidates = [
        c for c in df.columns
        if c != timestamp_col and c.lower().strip() in ('time', 'heure', 'hora')
    ]
    if col_lower in ('date', 'datum', 'fecha') and time_col_candidates:
        time_col = time_col_candidates[0]
        try:
            combined = df[timestamp_col].astype(str) + ' ' + df[time_col].astype(str)
            parsed_combined = pd.to_datetime(combined, errors='coerce')
            if parsed_combined.notna().sum() > len(df) * 0.9:
                df['_combined_ts'] = parsed_combined
                timestamp_col = '_combined_ts'
                warnings.append(
                    f"Combined 'Date' and '{time_col}' columns into a single datetime."
                )
        except Exception:
            pass

    try:
        parsed_ts, ts_warnings = _parse_timestamp_column(df[timestamp_col])
        warnings.extend(ts_warnings)
        df[timestamp_col] = parsed_ts

        null_ts = df[timestamp_col].isna().sum()
        if null_ts > len(df) * 0.5:
            raise ValueError(
                f"More than 50% of timestamps in '{timestamp_col}' could not be parsed."
            )
        if null_ts > 0:
            warnings.append(f"{null_ts} unparseable timestamps removed.")
            df = df.dropna(subset=[timestamp_col])
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            f"Could not parse '{timestamp_col}' as dates. Error: {exc}"
        ) from exc

    weather_cols = [c for c in WEATHER_FEATURE_COLS_CHECK if c in df.columns]
    df[consumption_col] = pd.to_numeric(df[consumption_col], errors='coerce')
    for col in weather_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    null_count = df[consumption_col].isna().sum()
    if null_count > 0:
        warnings.append(
            f"{null_count} non-numeric values found in consumption column and removed."
        )

    clean = df[[timestamp_col, consumption_col] + weather_cols].copy()
    clean = clean.rename(columns={
        timestamp_col: 'timestamp',
        consumption_col: 'consumption',
    })
    clean = clean.dropna(subset=['consumption'])
    clean = clean.set_index('timestamp')
    clean = clean.sort_index()

    dupes = clean.index.duplicated().sum()
    if dupes > 0:
        warnings.append(f"{dupes} duplicate timestamps found and averaged.")
        clean = clean.groupby(clean.index).mean(numeric_only=True)

    if len(clean) > 1:
        median_gap = pd.Series(clean.index).diff().dropna().median()
        gap_minutes = median_gap.total_seconds() / 60
    else:
        gap_minutes = 60

    if gap_minutes < 55:
        warnings.append(
            f"Sub-hourly data detected ({gap_minutes:.0f}-min intervals) and resampled to hourly."
        )
        clean = clean.resample('1h').mean()
    elif gap_minutes > 65:
        warnings.append(
            f"Data interval appears to be {gap_minutes:.0f} minutes and was normalised to hourly."
        )
        clean = clean.resample('1h').mean()

    full_index = pd.date_range(clean.index.min(), clean.index.max(), freq='1h', name='timestamp')
    missing_hours = len(full_index.difference(clean.index))
    if missing_hours > 0:
        warnings.append(f"{missing_hours} missing hourly timestamps were inserted and filled.")
    clean = clean.reindex(full_index)

    clean = clean.ffill().bfill()
    clean = clean.rename(columns={'consumption': 'use [kW]'})
    clean.index.name = 'time'

    return clean, warnings


def validate_for_model(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    issues: List[str] = []

    if len(df) < 336:
        issues.append(
            f"Only {len(df)} hourly readings found. At least 336 (14 days) are needed for reliable forecasting."
        )
        return False, issues

    date_range = df.index.max() - df.index.min()
    if date_range.days < 13:
        issues.append(
            f"Data spans only {date_range.days} days. At least 14 days are needed for reliable forecasting."
        )
        return False, issues

    return True, issues


def get_data_summary(df: pd.DataFrame) -> Dict:
    summary = {
        'rows': len(df),
        'days': (df.index.max() - df.index.min()).days,
        'start': df.index.min().strftime('%d %b %Y'),
        'end': df.index.max().strftime('%d %b %Y'),
        'mean_kwh': round(float(df['use [kW]'].mean()), 3),
        'peak_kwh': round(float(df['use [kW]'].max()), 3),
        'min_kwh': round(float(df['use [kW]'].min()), 3),
        'nulls': int(df['use [kW]'].isna().sum()),
    }
    weather_cols = [c for c in WEATHER_FEATURE_COLS_CHECK if c in df.columns]
    if weather_cols:
        summary['weather_cols'] = weather_cols
    return summary


def detect_and_fix_weather(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    notes: List[str] = []
    weather_present = [c for c in WEATHER_FEATURE_COLS_CHECK if c in df.columns]

    if not weather_present:
        return df, notes

    notes.append("Weather columns detected in your file and used directly.")

    for col in WEATHER_COLS_FAHRENHEIT:
        if col in df.columns:
            sample_max = df[col].dropna().max()
            if sample_max > 50:
                df[col] = (df[col] - 32) * 5 / 9
                notes.append(f"'{col}' appeared to be in Fahrenheit and was converted to Celsius.")

    return df, notes
