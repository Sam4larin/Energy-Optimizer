import sys
import json
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# Path setup
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.model import load_model, load_metadata, make_forecast
from src.optimizer import (
    generate_recommendations,
    get_daily_summary,
    APPLIANCES,
)

# Page config
st.set_page_config(
    page_title="Energy Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Design System
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

/* ── Global reset ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background-color: #050d1a !important;
    color: #e2e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 20% 20%, #0a1628 0%, #050d1a 60%) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #070f1f 0%, #050d1a 100%) !important;
    border-right: 1px solid rgba(59, 130, 246, 0.15) !important;
}

[data-testid="stSidebar"] * {
    color: #cbd5e1 !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stSidebar"] .stMarkdown h2 {
    color: #60a5fa !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 1rem !important;
    letter-spacing: 0.05em !important;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer { visibility: hidden; }
[data-testid="stSidebarCollapseButton"] button {
    background: transparent !important;
    border: none !important;
    width: 2rem !important;
    height: 2rem !important;
    font-size: 0 !important;
    color: transparent !important;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='20' height='20' viewBox='0 0 24 24' fill='none' stroke='%233b82f6' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='15 18 9 12 15 6'%3E%3C/polyline%3E%3C/svg%3E") !important;
    background-repeat: no-repeat !important;
    background-position: center !important;
    background-size: 1.1rem !important;
    cursor: pointer !important;
}
[data-testid="stSidebarCollapseButton"] button:hover {
    color: transparent !important;
    font-size: 0 !important;
    background-color: rgba(59,130,246,0.1) !important;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='20' height='20' viewBox='0 0 24 24' fill='none' stroke='%2360a5fa' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='15 18 9 12 15 6'%3E%3C/polyline%3E%3C/svg%3E") !important;
    background-repeat: no-repeat !important;
    background-position: center !important;
    background-size: 1.1rem !important;
}
[data-testid="stSidebarCollapseButton"] button * {
    color: transparent !important;
    font-size: 0 !important;
    display: none !important;
}
[data-testid="stDecoration"] { display: none; }

/* ── Main content padding ── */
.main .block-container {
    padding: 2rem 2.5rem !important;
    max-width: 1400px !important;
}

/* ── Page header ── */
.page-header {
    margin-bottom: 2rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid rgba(59, 130, 246, 0.2);
}

.page-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: -0.02em;
    margin: 0 0 0.3rem 0;
}

.page-title span {
    color: #3b82f6;
}

.page-subtitle {
    font-size: 0.9rem;
    color: #64748b;
    font-weight: 400;
    letter-spacing: 0.01em;
    margin: 0;
}

/* ── Section headers ── */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #3b82f6;
    margin: 0 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(59,130,246,0.3), transparent);
}

/* ── Metric cards ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}

.metric-card {
    background: linear-gradient(135deg, #0d1f3c 0%, #0a1628 100%);
    border: 1px solid rgba(59, 130, 246, 0.2);
    border-radius: 12px;
    padding: 1.3rem 1.5rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s ease;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #3b82f6, #60a5fa, transparent);
}

.metric-card:hover {
    border-color: rgba(59, 130, 246, 0.45);
}

.metric-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 0.5rem;
    font-family: 'Space Mono', monospace;
}

.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #f1f5f9;
    line-height: 1;
    margin-bottom: 0.4rem;
}

.metric-delta {
    font-size: 0.78rem;
    font-weight: 500;
}

.metric-delta.up { color: #f87171; }
.metric-delta.down { color: #34d399; }
.metric-delta.neutral { color: #60a5fa; }

/* ── Recommendation cards ── */
.rec-card {
    background: linear-gradient(135deg, #0d1f3c 0%, #0a1628 100%);
    border: 1px solid rgba(59, 130, 246, 0.18);
    border-left: 3px solid #3b82f6;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
    position: relative;
    transition: border-color 0.2s, transform 0.15s;
}

.rec-card:hover {
    border-color: rgba(59, 130, 246, 0.5);
    border-left-color: #60a5fa;
    transform: translateX(3px);
}

.rec-appliance {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    font-weight: 700;
    color: #60a5fa;
    letter-spacing: 0.05em;
    margin-bottom: 0.4rem;
    text-transform: uppercase;
}

.rec-message {
    font-size: 0.92rem;
    color: #cbd5e1;
    line-height: 1.5;
    margin-bottom: 0.35rem;
}

.rec-avoid {
    font-size: 0.8rem;
    color: #475569;
    line-height: 1.4;
}

.rec-saving {
    display: inline-block;
    margin-top: 0.5rem;
    font-size: 0.78rem;
    font-weight: 700;
    color: #34d399;
    background: rgba(52, 211, 153, 0.08);
    border: 1px solid rgba(52, 211, 153, 0.2);
    border-radius: 20px;
    padding: 0.2rem 0.7rem;
    font-family: 'Space Mono', monospace;
}

/* ── Sidebar elements ── */
.sidebar-stat {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.4rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    font-size: 0.82rem;
}

.sidebar-stat-label { color: #475569; }
.sidebar-stat-value {
    font-family: 'Space Mono', monospace;
    color: #60a5fa;
    font-size: 0.78rem;
}

/* ── Streamlit widget overrides ── */
[data-testid="stDateInput"] input,
[data-testid="stDateInput"] div {
    background: #0d1f3c !important;
    border-color: rgba(59,130,246,0.3) !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}

.stCheckbox label {
    color: #94a3b8 !important;
    font-size: 0.88rem !important;
}

.stCheckbox [data-testid="stCheckbox"] {
    accent-color: #3b82f6;
}

/* ── Plotly chart containers ── */
[data-testid="stPlotlyChart"] {
    border-radius: 12px;
    overflow: hidden;
}

/* ── Legend items ── */
.legend-row {
    display: flex;
    gap: 1.5rem;
    margin-top: 0.6rem;
    flex-wrap: wrap;
}

.legend-item {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.78rem;
    color: #64748b;
}

.legend-dot {
    width: 10px;
    height: 10px;
    border-radius: 2px;
    flex-shrink: 0;
}
</style>
""", unsafe_allow_html=True)

# Plotly dark theme defaults
PLOT_BG    = '#070f1f'
PAPER_BG   = '#070f1f'
GRID_COLOR = 'rgba(59,130,246,0.08)'
AXIS_COLOR = '#1e3a5f'
TEXT_COLOR = '#64748b'
BLUE_MAIN  = '#3b82f6'
BLUE_LIGHT = '#60a5fa'
RED_COLOR  = '#f87171'
GREEN_COLOR = '#34d399'
ORANGE_COLOR = '#fb923c'


def dark_layout(fig, height=380, show_legend=True):
    fig.update_layout(
        height=height,
        margin=dict(t=20, b=10, l=10, r=10),
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PAPER_BG,
        font=dict(family='DM Sans', color=TEXT_COLOR, size=11),
        legend=dict(
            orientation='h',
            yanchor='bottom', y=1.02,
            xanchor='right', x=1,
            bgcolor='rgba(0,0,0,0)',
            font=dict(color='#64748b', size=10),
        ) if show_legend else dict(visible=False),
        xaxis=dict(
            gridcolor=GRID_COLOR,
            linecolor=AXIS_COLOR,
            tickcolor=AXIS_COLOR,
            tickfont=dict(color=TEXT_COLOR, size=10),
            title_font=dict(color=TEXT_COLOR),
        ),
        yaxis=dict(
            gridcolor=GRID_COLOR,
            linecolor=AXIS_COLOR,
            tickcolor=AXIS_COLOR,
            tickfont=dict(color=TEXT_COLOR, size=10),
            title_font=dict(color=TEXT_COLOR),
        ),
        bargap=0.25,
    )
    return fig


# Cached loaders

@st.cache_resource
def get_model():
    return load_model()

@st.cache_data
def get_hourly_data():
    df = pd.read_csv(
        ROOT / 'data' / 'processed' / 'hourly_data.csv',
        index_col='time',
        parse_dates=True
    )
    return df

@st.cache_data
def get_metadata():
    return load_metadata()


# Sidebar

df       = get_hourly_data()
metadata = get_metadata()

min_date = (df.index.min() + pd.Timedelta(days=7)).date()
max_date = df.index.max().date()

with st.sidebar:
    st.markdown("## ⚡ ENERGY\nOPTIMIZER")
    st.markdown("---")

    st.markdown("**FORECAST DATE**")
    selected_date = st.date_input(
        label="Select forecast date",
        value=max_date,
        min_value=min_date,
        max_value=max_date,
    )

    st.markdown("---")
    st.markdown("**APPLIANCES**")
    st.caption("Select what you want to schedule.")

    selected_appliances = []
    for appliance, info in APPLIANCES.items():
        if info['flexible']:
            if st.checkbox(appliance, value=True):
                selected_appliances.append(appliance)

    st.markdown("---")
    st.markdown("**MODEL STATS**")
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

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("XGBoost model trained on 349 days of US smart meter data (2016).")


# Page header

date_str = pd.Timestamp(selected_date).strftime("%A, %B %d %Y")

st.markdown(f"""
<div class="page-header">
    <p class="page-title">⚡ Energy <span>Optimizer</span></p>
    <p class="page-subtitle">24-hour consumption forecast · {date_str}</p>
</div>
""", unsafe_allow_html=True)


# Generate forecast

model, feature_cols = get_model()
forecast_date       = pd.Timestamp(selected_date)
recent_data         = df[df.index < forecast_date].tail(200)

if len(recent_data) < 168:
    st.error("Not enough historical data before this date. Please select a later date.")
    st.stop()

with st.spinner("Generating forecast..."):
    forecast = make_forecast(
        recent_data=recent_data,
        forecast_date=forecast_date,
        model=model,
        feature_cols=feature_cols,
        forecast_hours=24,
    )
    summary = get_daily_summary(forecast)
    recs    = generate_recommendations(forecast, selected_appliances)


# Panel 1: Forecast chart

st.markdown('<p class="section-label">24-Hour Consumption Forecast</p>',
            unsafe_allow_html=True)

yhat        = forecast['yhat'].values
peak_hour   = summary['peak_hour']
trough_hour = summary['trough_hour']
hour_labels = [f"{h:02d}:00" for h in range(24)]

bar_colors = [
    RED_COLOR if i == peak_hour
    else GREEN_COLOR if i == trough_hour
    else BLUE_MAIN
    for i in range(24)
]

hist_avg = df.groupby(df.index.hour)['use [kW]'].mean().values

fig_forecast = go.Figure()

fig_forecast.add_trace(go.Bar(
    x=hour_labels,
    y=yhat,
    marker=dict(
        color=bar_colors,
        opacity=0.85,
        line=dict(width=0),
    ),
    name='Forecast',
    hovertemplate='<b>%{x}</b><br>Predicted: %{y:.3f} kW<extra></extra>',
))

fig_forecast.add_trace(go.Scatter(
    x=hour_labels,
    y=hist_avg,
    mode='lines',
    name='Historical avg',
    line=dict(color=ORANGE_COLOR, width=1.5, dash='dot'),
    hovertemplate='<b>%{x}</b><br>Historical avg: %{y:.3f} kW<extra></extra>',
))

dark_layout(fig_forecast, height=340)
fig_forecast.update_layout(
    xaxis_title=None,
    yaxis_title='kW',
)

st.plotly_chart(fig_forecast, use_container_width=True)

st.markdown("""
<div class="legend-row">
    <div class="legend-item">
        <div class="legend-dot" style="background:#f87171"></div>
        Peak demand hour
    </div>
    <div class="legend-item">
        <div class="legend-dot" style="background:#34d399"></div>
        Lowest demand window
    </div>
    <div class="legend-item">
        <div class="legend-dot" style="background:#fb923c; border-radius:50%"></div>
        Full-year hourly average
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# Panel 2: Metric cards

st.markdown('<p class="section-label">Daily Summary</p>',
            unsafe_allow_html=True)

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

if not selected_appliances:
    st.info("Select appliances in the sidebar to see recommendations.")
elif not recs:
    st.warning("No recommendations generated.")
else:
    col_a, col_b = st.columns(2)
    for i, rec in enumerate(recs):
        saving_html = ""
        if rec['saving'] > 0:
            saving_html = f'<span class="rec-saving">💰 Saves ~${rec["saving"]:.2f} vs peak</span>'

        card_html = f"""
        <div class="rec-card">
            <div class="rec-appliance">⚡ {rec['appliance']}</div>
            <div class="rec-message">{rec['message']}</div>
            <div class="rec-avoid">{rec['avoid_message']}</div>
            {saving_html}
        </div>
        """
        if i % 2 == 0:
            col_a.markdown(card_html, unsafe_allow_html=True)
        else:
            col_b.markdown(card_html, unsafe_allow_html=True)


st.markdown("<br>", unsafe_allow_html=True)


# Panel 4: Historical pattern

st.markdown('<p class="section-label">Your Home\'s Typical Daily Pattern</p>',
            unsafe_allow_html=True)

hist_by_hour = df.groupby(df.index.hour)['use [kW]'].agg(['mean', 'std'])
hours_int    = list(range(24))

fig_hist = go.Figure()

# Std band
upper = (hist_by_hour['mean'] + hist_by_hour['std']).values
lower = np.clip(hist_by_hour['mean'] - hist_by_hour['std'], 0, None).values

fig_hist.add_trace(go.Scatter(
    x=hours_int + hours_int[::-1],
    y=list(upper) + list(lower[::-1]),
    fill='toself',
    fillcolor='rgba(59,130,246,0.07)',
    line=dict(color='rgba(0,0,0,0)'),
    name='±1 std dev',
    hoverinfo='skip',
))

# Mean line with glow effect — two traces, one thick faded, one sharp
fig_hist.add_trace(go.Scatter(
    x=hours_int,
    y=hist_by_hour['mean'].values,
    mode='lines',
    line=dict(color='rgba(59,130,246,0.2)', width=8),
    showlegend=False,
    hoverinfo='skip',
))

fig_hist.add_trace(go.Scatter(
    x=hours_int,
    y=hist_by_hour['mean'].values,
    mode='lines+markers',
    name='Hourly average',
    line=dict(color=BLUE_LIGHT, width=2),
    marker=dict(size=5, color=BLUE_LIGHT, symbol='circle'),
    hovertemplate='<b>%{x}:00</b><br>Avg: %{y:.3f} kW<extra></extra>',
))

dark_layout(fig_hist, height=280)
fig_hist.update_layout(
    xaxis=dict(
        tickvals=hours_int,
        ticktext=hour_labels,
        title=None,
    ),
    yaxis_title='kW',
)

st.plotly_chart(fig_hist, use_container_width=True)

st.caption(
    "Average consumption by hour across all 349 days of 2016 data. "
    "Shaded band shows ±1 standard deviation."
)

# Footer
st.markdown("---")
st.markdown(
    '<p style="font-size:0.75rem;color:#1e3a5f;font-family:\'Space Mono\',monospace;">'
    'ENERGY OPTIMIZER · XGBoost · Streamlit · 24.6% MAPE on consumption hours &gt;0.3 kW'
    '</p>',
    unsafe_allow_html=True
)