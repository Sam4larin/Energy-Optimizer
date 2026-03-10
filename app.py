"""
app.py

Energy Optimizer — Streamlit Dashboard
Version 2: user-uploaded meter data, weather integration,
dynamic appliance picker, custom tariff inputs, on-the-fly retraining.
"""

import sys
from html import escape
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# Path setup
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.model import load_model, load_metadata, make_forecast, retrain_on_user_data
from src.optimizer import (
    generate_recommendations,
    get_daily_summary,
    get_appliance_categories,
    get_flexible_appliances,
    APPLIANCE_LIBRARY,
    DEFAULT_PEAK_RATE,
    DEFAULT_OFF_PEAK_RATE,
    DEFAULT_CURRENCY,
)
from src.ingestor import (
    detect_columns,
    get_all_columns,
    normalise_to_hourly,
    validate_for_model,
    get_data_summary,
    read_csv_smart,
)
from src.weather import (
    geocode,
    fetch_historical_weather,
    fetch_forecast_weather,
    merge_weather_with_meter,
)
from src.ui_components import (
    heat_map_timeline,
    consumption_gauge,
    savings_ticker,
    pulse_header,
    weather_card,
)
import streamlit.components.v1 as components

# Page config
st.set_page_config(
    page_title="Energy Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Design system
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background-color: #050d1a !important;
    color: #e2e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 20% 20%, #0a1628 0%, #050d1a 60%) !important;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #070f1f 0%, #050d1a 100%) !important;
    border-right: 1px solid rgba(59,130,246,0.15) !important;
}
[data-testid="stSidebar"] * {
    color: #cbd5e1 !important;
    font-family: 'DM Sans', sans-serif !important;
}
#MainMenu, footer { visibility: hidden; }

[data-testid="collapsedControl"] { display: flex !important; visibility: visible !important; }
[data-testid="stSidebarCollapseButton"] button {
    background: transparent !important; border: none !important;
    width: 2rem !important; height: 2rem !important;
    font-size: 0 !important; color: transparent !important;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='20' height='20' viewBox='0 0 24 24' fill='none' stroke='%233b82f6' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='15 18 9 12 15 6'%3E%3C/polyline%3E%3C/svg%3E") !important;
    background-repeat: no-repeat !important; background-position: center !important;
    background-size: 1.1rem !important; cursor: pointer !important;
}
[data-testid="stSidebarCollapseButton"] button:hover {
    color: transparent !important; font-size: 0 !important;
    background-color: rgba(59,130,246,0.1) !important;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='20' height='20' viewBox='0 0 24 24' fill='none' stroke='%2360a5fa' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='15 18 9 12 15 6'%3E%3C/polyline%3E%3C/svg%3E") !important;
    background-repeat: no-repeat !important; background-position: center !important;
    background-size: 1.1rem !important;
}
[data-testid="stSidebarCollapseButton"] button * {
    color: transparent !important; font-size: 0 !important; display: none !important;
}
section[data-testid="stSidebar"] { min-width: 300px !important; width: 300px !important; }
section[data-testid="stSidebar"] > div { padding: 1.5rem 1.2rem !important; }

.main .block-container { padding: 2rem 2.5rem !important; max-width: 1400px !important; }

.page-header { margin-bottom: 2rem; padding-bottom: 1.5rem; border-bottom: 1px solid rgba(59,130,246,0.2); }
.page-title { font-family: 'Space Mono', monospace; font-size: 1.8rem; font-weight: 700; color: #f1f5f9; margin: 0 0 0.3rem 0; }
.page-title span { color: #3b82f6; }
.page-subtitle { font-size: 0.9rem; color: #64748b; margin: 0; }

.section-label {
    font-family: 'Space Mono', monospace; font-size: 0.65rem; font-weight: 700;
    letter-spacing: 0.15em; text-transform: uppercase; color: #3b82f6;
    margin: 0 0 1rem 0; display: flex; align-items: center; gap: 0.5rem;
}
.section-label::after { content: ''; flex: 1; height: 1px; background: linear-gradient(90deg, rgba(59,130,246,0.3), transparent); }

.metric-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 1.5rem; }
.metric-card {
    background: linear-gradient(135deg, #0d1f3c 0%, #0a1628 100%);
    border: 1px solid rgba(59,130,246,0.2); border-radius: 12px;
    padding: 1.3rem 1.5rem; position: relative; overflow: hidden;
}
.metric-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #3b82f6, #60a5fa, transparent);
}
.metric-label { font-size: 0.7rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: #475569; margin-bottom: 0.5rem; font-family: 'Space Mono', monospace; }
.metric-value { font-family: 'Space Mono', monospace; font-size: 1.8rem; font-weight: 700; color: #f1f5f9; line-height: 1; margin-bottom: 0.4rem; }
.metric-delta { font-size: 0.78rem; font-weight: 500; }
.metric-delta.up   { color: #f87171; }
.metric-delta.down { color: #34d399; }
.metric-delta.neutral { color: #60a5fa; }

.rec-card {
    background: linear-gradient(135deg, #0d1f3c 0%, #0a1628 100%);
    border: 1px solid rgba(59,130,246,0.18); border-left: 3px solid #3b82f6;
    border-radius: 10px; padding: 1.2rem 1.4rem; margin-bottom: 0.8rem;
    transition: border-color 0.2s, transform 0.15s;
}
.rec-card:hover { border-color: rgba(59,130,246,0.5); border-left-color: #60a5fa; transform: translateX(3px); }
.rec-appliance { font-family: 'Space Mono', monospace; font-size: 0.8rem; font-weight: 700; color: #60a5fa; letter-spacing: 0.05em; margin-bottom: 0.4rem; text-transform: uppercase; }
.rec-message { font-size: 0.92rem; color: #cbd5e1; line-height: 1.5; margin-bottom: 0.35rem; }
.rec-avoid { font-size: 0.8rem; color: #475569; line-height: 1.4; }
.rec-saving { display: inline-block; margin-top: 0.5rem; font-size: 0.78rem; font-weight: 700; color: #34d399; background: rgba(52,211,153,0.08); border: 1px solid rgba(52,211,153,0.2); border-radius: 20px; padding: 0.2rem 0.7rem; font-family: 'Space Mono', monospace; }

.upload-zone {
    border: 2px dashed rgba(59,130,246,0.3); border-radius: 12px;
    padding: 2rem; text-align: center; background: rgba(13,31,60,0.4);
    margin-bottom: 1rem;
}
.upload-zone p { color: #475569; font-size: 0.88rem; margin: 0.3rem 0; }

.data-summary {
    background: linear-gradient(135deg, #0d1f3c 0%, #0a1628 100%);
    border: 1px solid rgba(52,211,153,0.2); border-left: 3px solid #34d399;
    border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 1rem;
}
.data-summary-title { font-family: 'Space Mono', monospace; font-size: 0.7rem; color: #34d399; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem; }
.data-stat { font-size: 0.82rem; color: #94a3b8; margin: 0.2rem 0; }
.data-stat span { color: #e2e8f0; font-weight: 600; }

.weather-badge {
    background: rgba(59,130,246,0.1); border: 1px solid rgba(59,130,246,0.25);
    border-radius: 8px; padding: 0.45rem 0.8rem; margin: 0.3rem 0 0.5rem 0;
    font-size: 0.8rem; color: #60a5fa;
}

.warning-card {
    background: rgba(251,146,60,0.08); border: 1px solid rgba(251,146,60,0.2);
    border-left: 3px solid #fb923c; border-radius: 10px;
    padding: 0.8rem 1rem; margin-bottom: 0.5rem;
    font-size: 0.82rem; color: #94a3b8;
}

.sidebar-divider { border: none; border-top: 1px solid rgba(59,130,246,0.1); margin: 1rem 0; }
.sidebar-stat { display: flex; justify-content: space-between; align-items: center; padding: 0.4rem 0; border-bottom: 1px solid rgba(255,255,255,0.04); font-size: 0.82rem; }
.sidebar-stat-label { color: #475569; }
.sidebar-stat-value { font-family: 'Space Mono', monospace; color: #60a5fa; font-size: 0.78rem; }

.legend-row { display: flex; gap: 1.5rem; margin-top: 0.6rem; flex-wrap: wrap; }
.legend-item { display: flex; align-items: center; gap: 0.4rem; font-size: 0.78rem; color: #64748b; }
.legend-dot { width: 10px; height: 10px; border-radius: 2px; flex-shrink: 0; }

.mode-badge {
    display: inline-block; font-family: 'Space Mono', monospace;
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; padding: 0.25rem 0.7rem;
    border-radius: 20px; margin-left: 0.5rem;
}
.mode-badge.demo { background: rgba(99,102,241,0.15); color: #818cf8; border: 1px solid rgba(99,102,241,0.3); }
.mode-badge.live { background: rgba(52,211,153,0.15); color: #34d399; border: 1px solid rgba(52,211,153,0.3); }

/* Category dropdown buttons */
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
    background: rgba(13,31,60,0.6) !important;
    border: 1px solid rgba(59,130,246,0.2) !important;
    border-radius: 6px !important;
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    text-align: left !important;
    padding: 0.45rem 0.8rem !important;
    margin-bottom: 0.25rem !important;
    transition: all 0.15s !important;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover {
    background: rgba(59,130,246,0.12) !important;
    border-color: rgba(59,130,246,0.4) !important;
    color: #e2e8f0 !important;
}
            /* Budget planner metric deltas */
[data-testid="stMetricDelta"] { font-size: 0.75rem !important; }
[data-testid="stMetricValue"] { font-family: 'Space Mono', monospace !important; color: #f1f5f9 !important; }
[data-testid="stMetricLabel"] { font-size: 0.7rem !important; color: #475569 !important; text-transform: uppercase; letter-spacing: 0.08em; }

            /* Smooth chart animations */
.js-plotly-plot .plotly .bars .point path {
    transition: all 0.3s ease !important;
}
/* Remove default Streamlit component borders */
iframe { border: none !important; }
/* Metric card hover */
.metric-card { transition: transform 0.2s, box-shadow 0.2s; }
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(59,130,246,0.15);
}
/* Rec card entrance animation */
@keyframes card-in {
    from { opacity: 0; transform: translateX(-8px); }
    to   { opacity: 1; transform: translateX(0); }
}
.rec-card { animation: card-in 0.3s ease both; }
</style>
""", unsafe_allow_html=True)

# Plot theme
PLOT_BG      = '#070f1f'
PAPER_BG     = '#070f1f'
GRID_COLOR   = 'rgba(59,130,246,0.08)'
AXIS_COLOR   = '#1e3a5f'
TEXT_COLOR   = '#64748b'
BLUE_MAIN    = '#3b82f6'
BLUE_LIGHT   = '#60a5fa'
RED_COLOR    = '#f87171'
GREEN_COLOR  = '#34d399'
ORANGE_COLOR = '#fb923c'


def dark_layout(fig, height=380, show_legend=True):
    fig.update_layout(
        height=height,
        margin=dict(t=20, b=10, l=10, r=10),
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        font=dict(family='DM Sans', color=TEXT_COLOR, size=11),
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02,
            xanchor='right', x=1, bgcolor='rgba(0,0,0,0)',
            font=dict(color='#64748b', size=10),
        ) if show_legend else dict(visible=False),
        xaxis=dict(gridcolor=GRID_COLOR, linecolor=AXIS_COLOR, tickcolor=AXIS_COLOR,
                   tickfont=dict(color=TEXT_COLOR, size=10), title_font=dict(color=TEXT_COLOR)),
        yaxis=dict(gridcolor=GRID_COLOR, linecolor=AXIS_COLOR, tickcolor=AXIS_COLOR,
                   tickfont=dict(color=TEXT_COLOR, size=10), title_font=dict(color=TEXT_COLOR)),
        bargap=0.25,
    )
    return fig


# Session state
if 'user_df'           not in st.session_state: st.session_state.user_df           = None
if 'user_model'        not in st.session_state: st.session_state.user_model        = None
if 'user_feat_cols'    not in st.session_state: st.session_state.user_feat_cols    = None
if 'user_metadata'     not in st.session_state: st.session_state.user_metadata     = None
if 'col_mapping'       not in st.session_state: st.session_state.col_mapping       = None
if 'raw_upload'        not in st.session_state: st.session_state.raw_upload        = None
if 'custom_appliances' not in st.session_state: st.session_state.custom_appliances = []
if 'open_categories'   not in st.session_state: st.session_state.open_categories   = {"Laundry", "Transport"}
if 'show_custom_form'  not in st.session_state: st.session_state.show_custom_form  = False
if 'user_location'     not in st.session_state: st.session_state.user_location     = None
if 'user_lat'          not in st.session_state: st.session_state.user_lat          = None
if 'user_lon'          not in st.session_state: st.session_state.user_lon          = None
if 'weather_df'        not in st.session_state: st.session_state.weather_df        = None


# Cached loaders
@st.cache_resource
def get_demo_model():
    return load_model()

@st.cache_data
def get_demo_data():
    return pd.read_csv(
        ROOT / 'data' / 'processed' / 'hourly_data.csv',
        index_col='time', parse_dates=True
    )

@st.cache_data
def get_demo_metadata():
    return load_metadata()


# Mode
is_user_mode = st.session_state.user_model is not None


# SIDEBAR

with st.sidebar:

    st.markdown("## ⚡ ENERGY OPTIMIZER")
    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # 1. LOCATION
    st.markdown("**YOUR LOCATION**")
    st.caption("Enter your city so we can fetch real weather for your forecasts.")

    loc_col1, loc_col2 = st.columns([3, 1])
    with loc_col1:
        location_input = st.text_input(
            "location",
            value=st.session_state.user_location or "",
            placeholder="e.g. Lagos, London, 90210",
            label_visibility="collapsed",
            key="location_input_field",
        )
    with loc_col2:
        fetch_wx_btn = st.button(
            "🌍",
            help="Fetch weather for this location",
            use_container_width=True,
            key="fetch_wx_btn",
        )

    if fetch_wx_btn:
        if location_input.strip():
            with st.spinner("Finding location..."):
                try:
                    lat, lon, resolved = geocode(location_input.strip())
                    st.session_state.user_lat      = lat
                    st.session_state.user_lon      = lon
                    st.session_state.user_location = resolved

                    # If meter data already loaded, refresh historical weather
                    if st.session_state.user_df is not None:
                        start_str = st.session_state.user_df.index.min().strftime('%Y-%m-%d')
                        end_str   = st.session_state.user_df.index.max().strftime('%Y-%m-%d')
                        with st.spinner(f"Fetching weather for {resolved}..."):
                            wx = fetch_historical_weather(lat, lon, start_str, end_str)
                            st.session_state.weather_df = wx
                        st.success(f"✅ Weather updated for {resolved}")
                    else:
                        st.success(f"✅ {resolved} — upload meter data to use weather")
                except Exception as e:
                    st.error(f"Location error: {e}")
        else:
            st.warning("Enter a city or postcode first.")

    if st.session_state.user_location:
        st.markdown(
            f'<div class="weather-badge">📍 {escape(st.session_state.user_location)}</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # 2. MODE INDICATOR
    if is_user_mode:
        st.markdown(
            '🟢 **Your data** <span class="mode-badge live">PERSONALISED</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '🔵 **Demo data** <span class="mode-badge demo">2016 DATASET</span>',
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # 3. METER DATA UPLOAD
    st.markdown("**YOUR METER DATA**")
    st.caption("Upload a CSV from your smart meter or energy monitor.")

    uploaded_file = st.file_uploader(
        label="Upload CSV",
        type=["csv"],
        label_visibility="collapsed",
        help="Any CSV with a timestamp column and an energy consumption column.",
    )

    if uploaded_file is not None:
        if st.session_state.raw_upload != uploaded_file.name:
            # New file — reset data but keep location
            st.session_state.raw_upload     = uploaded_file.name
            st.session_state.user_df        = None
            st.session_state.user_model     = None
            st.session_state.user_feat_cols = None
            st.session_state.user_metadata  = None
            st.session_state.col_mapping    = None
            st.session_state.weather_df     = None

            try:
                raw_df, read_warnings = read_csv_smart(uploaded_file)
                for rw in read_warnings:
                    st.markdown(f'<div class="warning-card">ℹ️ {rw}</div>',
                                unsafe_allow_html=True)
                detected = detect_columns(raw_df)
                all_cols = get_all_columns(raw_df)

                if detected['timestamp_confident'] and detected['consumption_confident']:
                    st.session_state.col_mapping = {
                        'raw_df':    raw_df,
                        'ts_col':    detected['timestamp_col'],
                        'con_col':   detected['consumption_col'],
                        'confirmed': True,
                    }
                else:
                    st.session_state.col_mapping = {
                        'raw_df':    raw_df,
                        'ts_col':    detected['timestamp_col'],
                        'con_col':   detected['consumption_col'],
                        'all_cols':  all_cols,
                        'confirmed': False,
                    }
            except Exception as e:
                st.error(f"Could not read file: {e}")

    # Column confirmation
    if (st.session_state.col_mapping is not None
            and not st.session_state.col_mapping.get('confirmed', False)):

        mapping  = st.session_state.col_mapping
        all_cols = mapping['all_cols']
        st.caption("⚠️ Please confirm which columns to use:")

        ts_default  = all_cols.index(mapping['ts_col'])  if mapping['ts_col']  in all_cols else 0
        con_default = all_cols.index(mapping['con_col']) if mapping['con_col'] in all_cols else 0

        chosen_ts  = st.selectbox("Timestamp column",   all_cols, index=ts_default)
        chosen_con = st.selectbox("Consumption column", all_cols, index=con_default)

        if st.button("✅ Confirm columns", use_container_width=True):
            st.session_state.col_mapping['ts_col']    = chosen_ts
            st.session_state.col_mapping['con_col']   = chosen_con
            st.session_state.col_mapping['confirmed'] = True
            st.rerun()

    # Weather prompt (columns confirmed, no weather yet, no location set)
    if (st.session_state.col_mapping is not None
            and st.session_state.col_mapping.get('confirmed', False)
            and st.session_state.user_model is None
            and st.session_state.weather_df is None
            and st.session_state.user_lat is None):

        st.markdown(
            '<div class="warning-card">'
            '💡 Enter your city above and click 🌍 to add real weather to your forecast. '
            'Or click below to train without it.'
            '</div>',
            unsafe_allow_html=True,
        )
        if st.button("⏭ Train without weather", use_container_width=True):
            st.session_state.weather_df = pd.DataFrame()  # empty = skipped
            st.rerun()

    # Auto-fetch weather if location is set but weather not yet fetched
    if (st.session_state.col_mapping is not None
            and st.session_state.col_mapping.get('confirmed', False)
            and st.session_state.user_model is None
            and st.session_state.weather_df is None
            and st.session_state.user_lat is not None):

        # Location already set — auto-fetch weather silently then proceed
        mapping = st.session_state.col_mapping
        try:
            hourly_tmp, _ = normalise_to_hourly(
                mapping['raw_df'], mapping['ts_col'], mapping['con_col']
            )
            start_str = hourly_tmp.index.min().strftime('%Y-%m-%d')
            end_str   = hourly_tmp.index.max().strftime('%Y-%m-%d')
            with st.spinner(f"Fetching weather for {st.session_state.user_location}..."):
                wx = fetch_historical_weather(
                    st.session_state.user_lat,
                    st.session_state.user_lon,
                    start_str, end_str,
                )
                st.session_state.weather_df = wx
            st.rerun()
        except Exception:
            st.session_state.weather_df = pd.DataFrame()
            st.rerun()

    # Retrain once columns confirmed AND weather resolved
    if (st.session_state.col_mapping is not None
            and st.session_state.col_mapping.get('confirmed', False)
            and st.session_state.user_model is None
            and st.session_state.weather_df is not None):

        mapping = st.session_state.col_mapping
        raw_df  = mapping['raw_df']
        ts_col  = mapping['ts_col']
        con_col = mapping['con_col']

        try:
            hourly_df, warnings = normalise_to_hourly(raw_df, ts_col, con_col)
            is_valid, issues    = validate_for_model(hourly_df)

            for w in warnings:
                st.markdown(f'<div class="warning-card">⚠️ {w}</div>',
                            unsafe_allow_html=True)

            if not is_valid:
                for issue in issues:
                    st.error(issue)
            else:
                if issues:
                    for issue in issues:
                        st.markdown(f'<div class="warning-card">⚠️ {issue}</div>',
                                    unsafe_allow_html=True)

                progress_text = st.empty()

                def update_progress(msg: str):
                    progress_text.caption(f"⚙️ {msg}")

                # Check if uploaded file already contains weather columns
                from src.ingestor import detect_and_fix_weather
                hourly_df, wx_notes = detect_and_fix_weather(hourly_df)
                for note in wx_notes:
                    st.markdown(f'<div class="warning-card">ℹ️ {note}</div>',
                                unsafe_allow_html=True)

                # If no built-in weather, merge fetched weather if available
                has_weather = any(c in hourly_df.columns for c in ['temperature', 'humidity'])
                if not has_weather and len(st.session_state.weather_df) > 0:
                    update_progress("Merging weather data...")
                    hourly_df = merge_weather_with_meter(
                        hourly_df, st.session_state.weather_df
                    )
                elif st.session_state.user_lat is not None:
                    # Weather not fetched yet despite location being set — fetch now
                    try:
                        update_progress("Fetching weather for your location...")
                        start_str = hourly_df.index.min().strftime('%Y-%m-%d')
                        end_str   = hourly_df.index.max().strftime('%Y-%m-%d')
                        wx = fetch_historical_weather(
                            st.session_state.user_lat,
                            st.session_state.user_lon,
                            start_str, end_str,
                        )
                        st.session_state.weather_df = wx
                        hourly_df = merge_weather_with_meter(hourly_df, wx)
                    except Exception:
                        pass  # Train without weather if fetch fails

                with st.spinner("Training model on your data..."):
                    model, feat_cols, metadata = retrain_on_user_data(
                        hourly_df,
                        progress_callback=update_progress,
                    )

                progress_text.empty()

                st.session_state.user_df        = hourly_df
                st.session_state.user_model     = model
                st.session_state.user_feat_cols = feat_cols
                st.session_state.user_metadata  = metadata
                st.rerun()

        except Exception as e:
            st.error(f"Processing failed: {e}")

    # Clear data button
    if is_user_mode:
        if st.button("🗑 Clear my data / use demo", use_container_width=True):
            for key in ['user_df', 'user_model', 'user_feat_cols', 'user_metadata',
                        'col_mapping', 'raw_upload', 'weather_df']:
                st.session_state[key] = None
            # Keep location — user doesn't need to re-enter it
            st.rerun()

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # 4. FORECAST DATE
    st.markdown("**FORECAST DATE**")
    active_df = st.session_state.user_df if is_user_mode else get_demo_data()
    min_date  = (active_df.index.min() + pd.Timedelta(days=7)).date()
    max_date  = (active_df.index.max() + pd.Timedelta(days=1)).date()

    selected_date = st.date_input(
        label="Select forecast date",
        value=max_date,
        min_value=min_date,
        max_value=max_date,
    )

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # 5. ELECTRICITY TARIFF
    st.markdown("**ELECTRICITY TARIFF**")
    flat_rate = st.toggle("I'm on a flat rate", value=True)

    col_c, col_p = st.columns([1, 2])
    with col_c:
        currency = st.text_input("Currency", value=DEFAULT_CURRENCY, max_chars=2)
    with col_p:
        if flat_rate:
            single_rate   = st.number_input(
                "Rate (per kWh)", value=DEFAULT_PEAK_RATE,
                min_value=0.0, max_value=200.0, step=0.5, format="%.1f",
            )
            peak_rate     = single_rate
            off_peak_rate = single_rate
            st.caption("Savings show demand reduction % instead of cost.")
        else:
            peak_rate = st.number_input(
                "Peak rate (per kWh)", value=DEFAULT_PEAK_RATE,
                min_value=0.0, max_value=200.0, step=0.5, format="%.1f",
            )
            off_peak_rate = st.number_input(
                "Off-peak rate (per kWh)", value=DEFAULT_OFF_PEAK_RATE,
                min_value=0.0, max_value=200.0, step=0.5, format="%.1f",
            )

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # 6. APPLIANCE PICKER
    st.markdown("**YOUR APPLIANCES**")
    st.caption("Select what you want to schedule.")

    categories               = get_appliance_categories()
    selected_appliance_names = []

    for category, appliance_names in categories.items():
        flexible_in_cat = [a for a in appliance_names if APPLIANCE_LIBRARY[a]['flexible']]
        if not flexible_in_cat:
            continue

        is_open = category in st.session_state.open_categories
        arrow   = "▾" if is_open else "▸"

        if st.button(f"{arrow}  {category}", key=f"cat_{category}", use_container_width=True):
            if is_open:
                st.session_state.open_categories.discard(category)
            else:
                st.session_state.open_categories.add(category)
            st.rerun()

        if is_open:
            for name in flexible_in_cat:
                info  = APPLIANCE_LIBRARY[name]
                label = f"{name} ({info['wattage']}W · {info['run_hours']}h)"
                if st.checkbox(label, value=False, key=f"app_{name}"):
                    selected_appliance_names.append(name)

    # Custom appliance
    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown("**ADD CUSTOM APPLIANCE**")

    custom_arrow = "▾" if st.session_state.show_custom_form else "▸"
    if st.button(
        f"{custom_arrow}  Add an appliance not listed",
        use_container_width=True,
        key="toggle_custom",
    ):
        st.session_state.show_custom_form = not st.session_state.show_custom_form
        st.rerun()

    if st.session_state.show_custom_form:
        custom_name    = st.text_input("Appliance name",   key="custom_name")
        custom_wattage = st.number_input("Wattage (W)",    min_value=1,   max_value=20000, value=1000,          key="custom_wattage")
        custom_hours   = st.number_input("Run time (hrs)", min_value=0.1, max_value=24.0,  value=1.0, step=0.1, key="custom_hours")
        if st.button("✚ Add appliance", use_container_width=True, key="add_custom_btn"):
            if custom_name.strip():
                st.session_state.custom_appliances.append({
                    'name':      custom_name.strip(),
                    'wattage':   custom_wattage,
                    'run_hours': custom_hours,
                })
                st.session_state.show_custom_form = False
                st.rerun()

    if st.session_state.custom_appliances:
        st.caption("Your custom appliances:")
        to_remove = []
        for i, ca in enumerate(st.session_state.custom_appliances):
            col1, col2 = st.columns([3, 1])
            col1.caption(f"⚡ {ca['name']} ({ca['wattage']}W)")
            if col2.button("✕", key=f"remove_{i}"):
                to_remove.append(i)
        for i in reversed(to_remove):
            st.session_state.custom_appliances.pop(i)
        if to_remove:
            st.rerun()

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # 7. MODEL STATS
    st.markdown("**MODEL STATS**")
    metadata = st.session_state.user_metadata if is_user_mode else get_demo_metadata()

    wx_trained = (
        st.session_state.weather_df is not None
        and len(st.session_state.weather_df) > 0
    )

    st.markdown(f"""
    <div class="sidebar-stat">
        <span class="sidebar-stat-label">MAPE</span>
        <span class="sidebar-stat-value">{metadata['model_mape']}%</span>
    </div>
    <div class="sidebar-stat">
        <span class="sidebar-stat-label">Baseline</span>
        <span class="sidebar-stat-value">{metadata['baseline_mape']}%</span>
    </div>
    <div class="sidebar-stat">
        <span class="sidebar-stat-label">MAE</span>
        <span class="sidebar-stat-value">{metadata['mae']} kW</span>
    </div>
    """, unsafe_allow_html=True)

    if is_user_mode:
        md = st.session_state.user_metadata
        st.markdown(f"""
        <div class="sidebar-stat">
            <span class="sidebar-stat-label">Data days</span>
            <span class="sidebar-stat-value">{md.get('data_days', 'N/A')}</span>
        </div>
        <div class="sidebar-stat">
            <span class="sidebar-stat-label">Train rows</span>
            <span class="sidebar-stat-value">{md.get('train_rows', 'N/A')}</span>
        </div>
        <div class="sidebar-stat">
            <span class="sidebar-stat-label">Weather</span>
            <span class="sidebar-stat-value">{'✅ Yes' if wx_trained else '— No'}</span>
        </div>
        """, unsafe_allow_html=True)


# MAIN CONTENT

date_str   = pd.Timestamp(selected_date).strftime("%A, %B %d %Y")
mode_badge = (
    '<span class="mode-badge live">YOUR DATA</span>'
    if is_user_mode else
    '<span class="mode-badge demo">DEMO</span>'
)

components.html(
    pulse_header(
        location=escape(st.session_state.user_location) if st.session_state.user_location else None,
        is_user_mode=is_user_mode,
    ),
    height=140,
)

# Weather card — shown in user mode when forecast weather is available
if is_user_mode and st.session_state.user_lat is not None:
    try:
        from datetime import date
        wx_today = fetch_forecast_weather(
            st.session_state.user_lat,
            st.session_state.user_lon,
            date.today().strftime('%Y-%m-%d'),
        )
        if len(wx_today) > 0:
            components.html(
                weather_card(
                    wx_today,
                    location=escape(st.session_state.user_location) if st.session_state.user_location else None,
                ),
                height=110,
                scrolling=False,
            )
    except Exception:
        pass  # Silently skip if weather fetch fails

# Upload prompt
if not is_user_mode and uploaded_file is None:
    st.markdown("""
    <div class="upload-zone">
        <p style="font-size:1.1rem;color:#60a5fa;font-weight:600;">
            📂 Upload your smart meter data for a personalised forecast
        </p>
        <p>Upload a CSV file from your energy provider, smart meter, or home monitor.</p>
        <p>The app will auto-detect your columns, fetch real weather for your location,
        retrain the model on your data, and generate forecasts specific to your home.</p>
        <p style="margin-top:0.8rem;color:#3b82f6;">
            No data yet? The demo below uses a real US household from 2016.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Data summary (user mode)
if is_user_mode:
    ss        = get_data_summary(st.session_state.user_df)
    wx_status = (
        f"✅ {escape(st.session_state.user_location)}"
        if st.session_state.weather_df is not None and len(st.session_state.weather_df) > 0
        else "Not used — forecast uses time & lag features only"
    )
    st.markdown(f"""
    <div class="data-summary">
        <div class="data-summary-title">✅ Your Data Loaded</div>
        <div class="data-stat">Period: <span>{ss['start']} → {ss['end']}</span></div>
        <div class="data-stat">Coverage: <span>{ss['days']} days · {ss['rows']:,} hourly readings</span></div>
        <div class="data-stat">Average: <span>{ss['mean_kwh']} kW/h</span> · Peak: <span>{ss['peak_kwh']} kW</span></div>
        <div class="data-stat">Weather: <span>{wx_status}</span></div>
    </div>
    """, unsafe_allow_html=True)

# Load model and data
if is_user_mode:
    model     = st.session_state.user_model
    feat_cols = st.session_state.user_feat_cols
    df        = st.session_state.user_df
else:
    model, feat_cols = get_demo_model()
    df               = get_demo_data()

# Generate forecast
forecast_date = pd.Timestamp(selected_date)
recent_data   = df[df.index < forecast_date].tail(200)

if len(recent_data) < 168:
    st.error("Not enough historical data before this date. Please select a later date.")
    st.stop()

appliances = get_flexible_appliances(
    selected_appliance_names,
    custom=st.session_state.custom_appliances,
)

with st.spinner("Generating forecast..."):
    future_weather = None

    # Inject real forecast weather for user mode if location is set
    if is_user_mode and st.session_state.user_lat is not None:
        try:
            future_weather = fetch_forecast_weather(
                st.session_state.user_lat,
                st.session_state.user_lon,
                forecast_date.strftime('%Y-%m-%d'),
            )
        except Exception:
            future_weather = None

    forecast = make_forecast(
        recent_data=recent_data,
        forecast_date=forecast_date,
        model=model,
        feature_cols=feat_cols,
        forecast_hours=24,
        future_weather=future_weather,
    )
    summary = get_daily_summary(forecast)
    recs    = generate_recommendations(
        forecast, appliances,
        peak_rate=peak_rate,
        off_peak_rate=off_peak_rate,
        currency=currency,
        show_cost_savings=not flat_rate,
    )

# Panel 1: Forecast chart
st.markdown('<p class="section-label">24-Hour Consumption Forecast</p>',
            unsafe_allow_html=True)

yhat        = forecast['yhat'].values
peak_hour   = summary['peak_hour']
trough_hour = summary['trough_hour']
hour_labels = [f"{h:02d}:00" for h in range(24)]

bar_colors = [
    RED_COLOR   if i == peak_hour   else
    GREEN_COLOR if i == trough_hour else
    BLUE_MAIN
    for i in range(24)
]

hist_avg = df.groupby(df.index.hour)['use [kW]'].mean().values

fig_forecast = go.Figure()
fig_forecast.add_trace(go.Bar(
    x=hour_labels, y=yhat,
    marker=dict(color=bar_colors, opacity=0.85, line=dict(width=0)),
    name='Forecast',
    hovertemplate='<b>%{x}</b><br>Predicted: %{y:.3f} kW<extra></extra>',
))
fig_forecast.add_trace(go.Scatter(
    x=hour_labels, y=hist_avg, mode='lines', name='Historical avg',
    line=dict(color=ORANGE_COLOR, width=1.5, dash='dot'),
    hovertemplate='<b>%{x}</b><br>Historical avg: %{y:.3f} kW<extra></extra>',
))
dark_layout(fig_forecast, height=340)
fig_forecast.update_layout(xaxis_title=None, yaxis_title='kW')
st.plotly_chart(fig_forecast, use_container_width=True)

st.markdown("""
<div class="legend-row">
    <div class="legend-item"><div class="legend-dot" style="background:#f87171"></div>Peak demand hour</div>
    <div class="legend-item"><div class="legend-dot" style="background:#34d399"></div>Lowest demand window</div>
    <div class="legend-item"><div class="legend-dot" style="background:#fb923c;border-radius:50%"></div>Historical average</div>
</div>
""", unsafe_allow_html=True)
# Heat map timeline
components.html(
    heat_map_timeline(
        yhat=list(forecast['yhat'].values),
        peak_hour=peak_hour,
        trough_hour=trough_hour,
        currency=currency,
    ),
    height=185,
    scrolling=False,
)

# Consumption gauge
hist_daily_avg = float(df.groupby(df.index.date)['use [kW]'].sum().mean())
components.html(
    consumption_gauge(
        total_kwh=summary['total_kwh'],
        avg_kwh=hist_daily_avg,
        currency=currency,
        rate=peak_rate,
    ),
    height=210,
)

st.markdown("<br>", unsafe_allow_html=True)

# Panel 2: Metric cards
st.markdown('<p class="section-label">Daily Summary</p>', unsafe_allow_html=True)
st.markdown(f"""
<div class="metric-grid">
    <div class="metric-card">
        <div class="metric-label">Peak Hour</div>
        <div class="metric-value">{summary['peak_time']}</div>
        <div class="metric-delta up">↑ {summary['peak_value']:.2f} kW predicted</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Best Window</div>
        <div class="metric-value">{summary['trough_time']}</div>
        <div class="metric-delta down">↓ {summary['trough_value']:.2f} kW predicted</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Total Predicted</div>
        <div class="metric-value">{summary['total_kwh']:.1f} <small style="font-size:1rem;color:#475569">kWh</small></div>
        <div class="metric-delta neutral">~ {summary['mean_kwh']:.2f} kW average</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Panel 3: Recommendations
st.markdown('<p class="section-label">Appliance Recommendations</p>',
            unsafe_allow_html=True)

# Savings ticker
if recs and not flat_rate:
    total_savings = sum(r['saving'] for r in recs)
    if total_savings > 0:
        components.html(
            savings_ticker(total_savings=total_savings, currency=currency),
            height=90,
        )

if not appliances:
    st.markdown("""
    <div class="upload-zone">
        <p style="color:#475569;">Select appliances from the sidebar to see scheduling recommendations.</p>
        <p>Click a category to expand it and check the appliances you want to optimise.</p>
    </div>
    """, unsafe_allow_html=True)
elif not recs:
    st.warning("No recommendations generated.")
else:
    col_a, col_b = st.columns(2)
    for i, rec in enumerate(recs):
        saving_html = ""
        if not flat_rate and rec['saving'] > 0:
            saving_html = (
                f'<span class="rec-saving">'
                f'💰 Saves ~{rec["currency"]}{rec["saving"]:.2f} vs peak'
                f'</span>'
            )
        rec_appliance = escape(str(rec['appliance']))
        rec_message = escape(str(rec['message']))
        rec_avoid = escape(str(rec['avoid_message']))
        card_html = f"""
        <div class="rec-card">
            <div class="rec-appliance">⚡ {rec_appliance}</div>
            <div class="rec-message">{rec_message}</div>
            <div class="rec-avoid">{rec_avoid}</div>
            {saving_html}
        </div>"""
        if i % 2 == 0:
            col_a.markdown(card_html, unsafe_allow_html=True)
        else:
            col_b.markdown(card_html, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Panel 4: Budget Planner
st.markdown('<p class="section-label">Energy Budget Planner</p>',
            unsafe_allow_html=True)

st.markdown("""
<div style="background:linear-gradient(135deg,#0d1f3c,#0a1628);border:1px solid rgba(59,130,246,0.2);
border-left:3px solid #f59e0b;border-radius:10px;padding:1rem 1.4rem;margin-bottom:1rem;">
<p style="color:#fbbf24;font-size:0.82rem;font-weight:700;margin:0 0 0.3rem 0;
font-family:'Space Mono',monospace;text-transform:uppercase;letter-spacing:0.08em;">
💡 How much energy do you have and how long do you want it to last?</p>
<p style="color:#64748b;font-size:0.82rem;margin:0;">
Enter your budget below — the app will tell you if it's achievable and exactly what to cut.</p>
</div>
""", unsafe_allow_html=True)

budget_col1, budget_col2 = st.columns(2)

with budget_col1:
    budget_mode = st.radio(
        "I know my",
        ["kWh bought", "Amount spent"],
        horizontal=True,
        label_visibility="visible",
    )

with budget_col2:
    budget_days = st.number_input(
        "Days I want it to last",
        min_value=1, max_value=365, value=7, step=1,
    )

b_col1, b_col2, b_col3 = st.columns(3)

if budget_mode == "kWh bought":
    with b_col1:
        budget_kwh = st.number_input(
            "kWh purchased", min_value=0.1, max_value=10000.0,
            value=50.0, step=1.0, format="%.1f",
        )
    with b_col2:
        budget_rate = st.number_input(
            f"Rate ({currency}/kWh)", min_value=0.0, max_value=500.0,
            value=float(peak_rate), step=0.5, format="%.2f",
        )
    with b_col3:
        st.markdown("<br>", unsafe_allow_html=True)
        run_budget = st.button("📊 Analyse budget", use_container_width=True)
    kwh_for_plan = budget_kwh

else:  # Amount spent
    with b_col1:
        budget_amount = st.number_input(
            f"Amount ({currency})", min_value=0.1, max_value=100000.0,
            value=500.0, step=10.0, format="%.2f",
        )
    with b_col2:
        budget_rate = st.number_input(
            f"Rate ({currency}/kWh)", min_value=0.01, max_value=500.0,
            value=float(peak_rate), step=0.5, format="%.2f",
        )
    with b_col3:
        kwh_derived = budget_amount / budget_rate if budget_rate > 0 else 0
        st.metric("kWh this buys", f"{kwh_derived:.1f} kWh")
        run_budget = st.button("📊 Analyse budget", use_container_width=True)
    kwh_for_plan = kwh_derived

if run_budget:
    if kwh_for_plan > 0 and budget_days > 0:
        from src.optimizer import calculate_budget_plan
        plan = calculate_budget_plan(
            historical_df=df,
            kwh_budget=kwh_for_plan,
            target_days=int(budget_days),
            rate_per_kwh=budget_rate,
            currency=currency,
            forecast=forecast,
            user_appliances=appliances if appliances else None,
        )

        # Verdict banner
        verdict_color = "#34d399" if plan['achievable'] else ("#f59e0b" if plan['stretch'] else "#f87171")
        st.markdown(f"""
        <div style="background:rgba(0,0,0,0.3);border:1px solid {verdict_color};
        border-left:4px solid {verdict_color};border-radius:10px;
        padding:1rem 1.4rem;margin:1rem 0;">
        <p style="color:{verdict_color};font-size:0.95rem;font-weight:600;margin:0;">{plan['verdict']}</p>
        </div>
        """, unsafe_allow_html=True)

        # Summary numbers
        s_col1, s_col2, s_col3, s_col4 = st.columns(4)
        s_col1.metric("Your daily budget", f"{plan['budget_daily_kwh']} kWh")
        s_col2.metric("Your current daily avg", f"{plan['avg_daily_kwh']} kWh")
        s_col3.metric("Daily cost at budget", f"{currency}{plan['budget_daily_cost']:.2f}")
        s_col4.metric(
            "At current usage lasts",
            f"{plan['days_at_current']:.1f} days",
            delta=f"{plan['days_at_current'] - budget_days:.1f} vs target",
        )

        st.markdown("<br>", unsafe_allow_html=True)

        left_col, right_col = st.columns(2)

        with left_col:
            st.markdown(
                '<p style="font-size:0.72rem;font-weight:700;letter-spacing:0.1em;'
                'text-transform:uppercase;color:#f59e0b;margin-bottom:0.6rem;">'
                '⏰ Your Most Expensive Hours</p>',
                unsafe_allow_html=True,
            )
            for h in plan['expensive_hours']:
                st.markdown(f"""
                <div style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.2);
                border-radius:8px;padding:0.6rem 0.9rem;margin-bottom:0.4rem;">
                <span style="color:#fbbf24;font-family:'Space Mono',monospace;font-size:0.82rem;
                font-weight:700;">{h['time']}</span>
                <span style="color:#64748b;font-size:0.8rem;margin-left:0.8rem;">
                {h['kwh']} kWh avg · {h['currency']}{h['cost']:.3f}/hr</span>
                </div>
                """, unsafe_allow_html=True)

        with right_col:
            st.markdown(
                '<p style="font-size:0.72rem;font-weight:700;letter-spacing:0.1em;'
                'text-transform:uppercase;color:#34d399;margin-bottom:0.6rem;">'
                '✂️ Specific Cuts to Hit Your Target</p>',
                unsafe_allow_html=True,
            )
            if plan['cuts']:
                for cut in plan['cuts']:
                    st.markdown(f"""
                    <div style="background:rgba(52,211,153,0.06);border:1px solid rgba(52,211,153,0.15);
                    border-radius:8px;padding:0.6rem 0.9rem;margin-bottom:0.4rem;">
                    <span style="color:#34d399;font-size:0.85rem;font-weight:600;">
                    {cut['appliance']}</span>
                    <span style="color:#64748b;font-size:0.8rem;"> — {cut['action']}</span><br>
                    <span style="color:#94a3b8;font-size:0.78rem;">
                    Saves ~{cut['daily_saving_kwh']} kWh/day
                    ({cut['currency']}{cut['daily_saving_cost']:.2f}/day)</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(
                    '<p style="color:#34d399;font-size:0.85rem;">'
                    'Your budget already covers current usage — no cuts needed.</p>',
                    unsafe_allow_html=True,
                )

        # Appliance breakdown bar chart
        st.markdown("<br>", unsafe_allow_html=True)

        if plan['is_personalised']:
            chart_label = 'Your Appliances — Cost Per Day'
            chart_note = (
                "Based on <b>your selected appliances</b>. "
                "Estimates assume each appliance runs for its typical cycle duration each day. "
                "🟢 Green = flexible &nbsp;🔴 Red = fixed &nbsp;⭐ = your selection"
            )
        else:
            chart_label = 'Typical Appliance Cost Per Day'
            chart_note = (
                "Generic estimates — <b>select appliances in the sidebar</b> "
                "to see figures for your home. "
                "🟢 Green = flexible &nbsp;🔴 Red = fixed"
            )

        st.markdown(
            f'<p style="font-size:0.72rem;font-weight:700;letter-spacing:0.1em;'
            f'text-transform:uppercase;color:#3b82f6;margin-bottom:0.6rem;">'
            f'{chart_label}</p>',
            unsafe_allow_html=True,
        )

        impacts = plan['appliance_impacts']

        # Mark user-selected appliances with a star in the y-axis labels
        bar_labels = [
            f"⭐ {a['name']}" if a.get('user_selected') else a['name']
            for a in impacts
        ]
        bar_colors = [
            '#34d399' if a['flexible'] else '#f87171'
            for a in impacts
        ]
        # Make user-selected bars slightly brighter
        bar_opacity = [
            1.0 if a.get('user_selected') else 0.65
            for a in impacts
        ]

        fig_budget = go.Figure(go.Bar(
            x=[a['daily_cost'] for a in impacts],
            y=bar_labels,
            orientation='h',
            marker=dict(
                color=bar_colors,
                opacity=bar_opacity,
            ),
            hovertemplate=(
                '<b>%{y}</b><br>'
                f'Daily cost: {currency}%{{x:.3f}}<br>'
                'kWh: %{customdata}<extra></extra>'
            ),
            customdata=[a['daily_kwh'] for a in impacts],
        ))

        # Budget line
        fig_budget.add_vline(
            x=plan['budget_daily_cost'],
            line_dash='dash', line_color='#f59e0b', line_width=2,
            annotation_text=f"Daily budget ({currency}{plan['budget_daily_cost']:.2f})",
            annotation_font_color='#fbbf24',
            annotation_font_size=10,
        )

        dark_layout(fig_budget, height=max(280, len(impacts) * 34), show_legend=False)
        fig_budget.update_layout(
            xaxis_title=f"Cost per day ({currency})",
            yaxis_title=None,
            margin=dict(l=150, r=20, t=20, b=30),
        )
        st.plotly_chart(fig_budget, use_container_width=True)
        st.markdown(
            f'<p style="font-size:0.75rem;color:#64748b;">{chart_note}</p>',
            unsafe_allow_html=True,
        )
    else:
        st.warning("Enter a valid kWh amount and number of days.")

# Panel 6: Historical pattern
st.markdown("<p class=\"section-label\">Your Home's Typical Daily Pattern</p>",
            unsafe_allow_html=True)

hist_by_hour = df.groupby(df.index.hour)['use [kW]'].agg(['mean', 'std'])
hours_int    = list(range(24))
upper        = (hist_by_hour['mean'] + hist_by_hour['std']).values
lower        = np.clip(hist_by_hour['mean'] - hist_by_hour['std'], 0, None).values

fig_hist = go.Figure()
fig_hist.add_trace(go.Scatter(
    x=hours_int + hours_int[::-1], y=list(upper) + list(lower[::-1]),
    fill='toself', fillcolor='rgba(59,130,246,0.07)',
    line=dict(color='rgba(0,0,0,0)'), name='±1 std dev', hoverinfo='skip',
))
fig_hist.add_trace(go.Scatter(
    x=hours_int, y=hist_by_hour['mean'].values, mode='lines',
    line=dict(color='rgba(59,130,246,0.2)', width=8),
    showlegend=False, hoverinfo='skip',
))
fig_hist.add_trace(go.Scatter(
    x=hours_int, y=hist_by_hour['mean'].values, mode='lines+markers',
    name='Hourly average', line=dict(color=BLUE_LIGHT, width=2),
    marker=dict(size=5, color=BLUE_LIGHT),
    hovertemplate='<b>%{x}:00</b><br>Avg: %{y:.3f} kW<extra></extra>',
))
dark_layout(fig_hist, height=280)
fig_hist.update_layout(
    xaxis=dict(tickvals=hours_int, ticktext=hour_labels, title=None),
    yaxis_title='kW',
)
st.plotly_chart(fig_hist, use_container_width=True)
st.caption(
    "Average consumption by hour across your full dataset. "
    "Shaded band shows ±1 standard deviation."
)

# Footer
st.markdown("---")
source = "your data" if is_user_mode else "2016 US smart meter demo data"
st.markdown(
    f'<p style="font-size:0.75rem;color:#1e3a5f;font-family:\'Space Mono\',monospace;">'
    f'ENERGY OPTIMIZER v2 · XGBoost · Streamlit · Open-Meteo · {source}'
    f'</p>',
    unsafe_allow_html=True,
)