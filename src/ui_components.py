"""
src/ui_components.py

Interactive HTML/JS components for the Energy Optimizer dashboard.
Injected via st.components.v1.html() for things Streamlit can't do natively:
- Animated counters
- Heat map timeline
- Consumption gauge
- Live weather card
- Animated savings ticker
"""


def heat_map_timeline(yhat: list, peak_hour: int, trough_hour: int, currency: str = "£") -> str:
    """
    24-hour heat map strip. Each cell coloured from deep blue (low)
    to fiery orange/red (high). Hoverable with tooltip.
    """
    max_val = max(yhat) if yhat else 1
    min_val = min(yhat) if yhat else 0
    val_range = max(max_val - min_val, 0.001)

    cells = ""
    for h, v in enumerate(yhat):
        ratio     = (v - min_val) / val_range
        label     = f"{h:02d}:00"
        tip       = f"{h:02d}:00 — {v:.3f} kW"
        is_peak   = h == peak_hour
        is_trough = h == trough_hour

        if is_peak:
            bg = "linear-gradient(135deg,#ef4444,#f97316)"
            border = "2px solid #fbbf24"
            icon = "🔴"
        elif is_trough:
            bg = "linear-gradient(135deg,#059669,#34d399)"
            border = "2px solid #6ee7b7"
            icon = "🟢"
        else:
            r = int(15  + ratio * 220)
            g = int(31  + ratio * 30)
            b = int(60  + (1 - ratio) * 150)
            bg = f"rgb({r},{g},{b})"
            border = "1px solid rgba(255,255,255,0.05)"
            icon = ""

        cells += f"""
        <div class="hm-cell" title="{tip}" style="background:{bg};border:{border};">
            <div class="hm-icon">{icon}</div>
            <div class="hm-val">{v:.2f}</div>
            <div class="hm-lbl">{label}</div>
        </div>"""

    return f"""
    <style>
    *{{box-sizing:border-box;margin:0;padding:0;}}
    html,body{{background:transparent;overflow-x:hidden;overflow-y:visible;height:auto;}}
    .hm-wrap{{padding:0.4rem 0 0.8rem 0;font-family:'Space Mono',monospace;}}
    .hm-label{{font-size:0.6rem;font-weight:700;letter-spacing:0.12em;
        text-transform:uppercase;color:#3b82f6;margin-bottom:0.5rem;}}
    .hm-strip{{
        display:flex;gap:3px;
        overflow-x:auto;overflow-y:visible;
        padding-bottom:6px;padding-top:4px;
    }}
    .hm-strip::-webkit-scrollbar{{height:4px;}}
    .hm-strip::-webkit-scrollbar-track{{background:rgba(255,255,255,0.03);border-radius:2px;}}
    .hm-strip::-webkit-scrollbar-thumb{{background:rgba(59,130,246,0.4);border-radius:2px;}}
    .hm-cell{{
        flex:0 0 44px;height:68px;border-radius:6px;
        display:flex;flex-direction:column;
        align-items:center;justify-content:center;cursor:pointer;
        transition:transform 0.15s,box-shadow 0.15s;position:relative;
    }}
    .hm-cell:hover{{
        transform:scaleY(1.12) translateY(-3px);
        box-shadow:0 8px 24px rgba(0,0,0,0.5);z-index:10;
    }}
    .hm-val{{font-size:0.58rem;color:rgba(255,255,255,0.92);font-weight:700;line-height:1.3;}}
    .hm-lbl{{font-size:0.52rem;color:rgba(255,255,255,0.45);margin-top:2px;}}
    .hm-icon{{font-size:0.65rem;line-height:1;}}
    .hm-legend{{display:flex;gap:1.4rem;margin-top:0.5rem;flex-wrap:wrap;}}
    .hm-legend span{{font-size:0.72rem;color:#64748b;white-space:nowrap;}}
    </style>
    <div class="hm-wrap">
        <div class="hm-label">⚡ 24-Hour Demand Heat Map</div>
        <div class="hm-strip">{cells}</div>
        <div class="hm-legend">
            <span>🔴 Peak demand</span>
            <span>🟢 Best window</span>
            <span>🔵 Lower demand</span>
        </div>
    </div>
    """


def consumption_gauge(total_kwh: float, avg_kwh: float, currency: str = "£", rate: float = 28.0) -> str:
    """
    SVG arc gauge showing today's predicted total vs historical average.
    """
    import math

    ratio    = min(total_kwh / max(avg_kwh, 0.001), 2.0)
    pct      = min(ratio * 50, 100)
    diff_pct = (total_kwh - avg_kwh) / max(avg_kwh, 0.001) * 100

    if ratio < 0.85:
        color  = "#34d399"; status = "Below average ✓"; glow = "rgba(52,211,153,0.4)"
    elif ratio < 1.15:
        color  = "#60a5fa"; status = "On track";         glow = "rgba(96,165,250,0.4)"
    elif ratio < 1.5:
        color  = "#f59e0b"; status = "Above average ⚠";  glow = "rgba(245,158,11,0.4)"
    else:
        color  = "#f87171"; status = "High usage 🔴";     glow = "rgba(248,113,113,0.4)"

    # Arc: centre at (100,100), radius 72, sweeps from -220° to +40° (260° total)
    cx, cy, r = 100, 100, 72
    start_deg = -220
    total_deg = 260
    sweep_deg = total_deg * (pct / 100)

    def pt(deg):
        rad = math.radians(deg)
        return cx + r * math.cos(rad), cy + r * math.sin(rad)

    def arc(d0, d1):
        x0, y0 = pt(d0)
        x1, y1 = pt(d1)
        large  = 1 if abs(d1 - d0) > 180 else 0
        return f"M{x0:.2f},{y0:.2f} A{r},{r} 0 {large},1 {x1:.2f},{y1:.2f}"

    bg_d   = arc(start_deg, start_deg + total_deg)
    fill_d = arc(start_deg, start_deg + max(sweep_deg, 0.5))

    sign   = "+" if diff_pct >= 0 else ""

    return f"""
    <style>
    *{{box-sizing:border-box;margin:0;padding:0;}}
    body{{background:transparent;overflow:hidden;font-family:'Space Mono',monospace;}}
    .gauge-wrap{{
        display:flex;align-items:center;gap:1.5rem;
        background:linear-gradient(135deg,#0d1f3c,#0a1628);
        border:1px solid rgba(59,130,246,0.2);border-radius:12px;
        padding:1rem 1.5rem;
    }}
    .gauge-stats div{{margin-bottom:0.45rem;}}
    .gauge-stats .lbl{{font-size:0.72rem;color:#64748b;font-family:sans-serif;}}
    .gauge-stats .val{{font-size:1rem;font-weight:700;}}
    .gauge-head{{font-size:0.6rem;text-transform:uppercase;letter-spacing:0.1em;
        color:#475569;margin-bottom:0.6rem;}}
    </style>
    <div class="gauge-wrap">
        <svg width="200" height="160" viewBox="0 0 200 160">
            <path d="{bg_d}" fill="none" stroke="rgba(255,255,255,0.06)"
                stroke-width="14" stroke-linecap="round"/>
            <path d="{fill_d}" fill="none" stroke="{color}"
                stroke-width="14" stroke-linecap="round"
                style="filter:drop-shadow(0 0 8px {glow});"/>
            <text x="100" y="95" text-anchor="middle" fill="#f1f5f9"
                font-family="Space Mono,monospace" font-size="20" font-weight="700">{total_kwh:.1f}</text>
            <text x="100" y="112" text-anchor="middle" fill="#475569"
                font-family="sans-serif" font-size="9.5">kWh today</text>
            <text x="100" y="128" text-anchor="middle" fill="{color}"
                font-family="sans-serif" font-size="9" font-weight="600">{status}</text>
        </svg>
        <div class="gauge-stats">
            <div class="gauge-head">Today vs Your Average</div>
            <div>
                <div class="lbl">Predicted total</div>
                <div class="val" style="color:#f1f5f9;">{total_kwh:.2f} kWh</div>
            </div>
            <div>
                <div class="lbl">Your daily average</div>
                <div class="val" style="color:#60a5fa;">{avg_kwh:.2f} kWh</div>
            </div>
            <div>
                <div class="lbl">vs average</div>
                <div class="val" style="color:{color};">{sign}{diff_pct:.1f}%</div>
            </div>
        </div>
    </div>
    """


def savings_ticker(total_savings: float, currency: str = "£") -> str:
    """
    Animated counter that counts up to total potential daily savings.
    """
    if total_savings <= 0:
        return ""

    return f"""
    <div style="background:linear-gradient(135deg,rgba(52,211,153,0.08),rgba(13,31,60,0.8));
    border:1px solid rgba(52,211,153,0.25);border-radius:12px;
    padding:1rem 1.5rem;margin-bottom:1rem;display:flex;
    align-items:center;justify-content:space-between;">
        <div>
            <div style="font-family:'Space Mono',monospace;font-size:0.62rem;font-weight:700;
            letter-spacing:0.15em;text-transform:uppercase;color:#34d399;margin-bottom:0.3rem;">
            💰 Total Potential Daily Savings</div>
            <div style="color:#64748b;font-size:0.8rem;">
            By running all selected appliances at optimal times</div>
        </div>
        <div style="text-align:right;">
            <div id="savings-counter" style="font-family:'Space Mono',monospace;
            font-size:2rem;font-weight:700;color:#34d399;
            text-shadow:0 0 20px rgba(52,211,153,0.5);">
            {currency}0.00</div>
            <div style="font-size:0.75rem;color:#475569;">per day if rescheduled</div>
        </div>
    </div>
    <script>
    (function() {{
        const target = {total_savings:.4f};
        const el = document.getElementById('savings-counter');
        if (!el) return;
        const sym = '{currency}';
        let start = null;
        const duration = 1200;
        function step(ts) {{
            if (!start) start = ts;
            const progress = Math.min((ts - start) / duration, 1);
            const ease = 1 - Math.pow(1 - progress, 3);
            el.textContent = sym + (target * ease).toFixed(2);
            if (progress < 1) requestAnimationFrame(step);
            else el.textContent = sym + target.toFixed(2);
        }}
        requestAnimationFrame(step);
    }})();
    </script>
    """


def pulse_header(location: str = None, is_user_mode: bool = False) -> str:
    """
    Animated header component with electricity pulse effect and live clock.
    """
    location_html = ""
    if location:
        location_html = f"""
        <div style="display:inline-flex;align-items:center;gap:0.4rem;
        background:rgba(59,130,246,0.1);border:1px solid rgba(59,130,246,0.2);
        border-radius:20px;padding:0.2rem 0.7rem;font-size:0.75rem;color:#60a5fa;">
        📍 {location}</div>"""

    mode_color = "#34d399" if is_user_mode else "#818cf8"
    mode_text  = "PERSONALISED" if is_user_mode else "DEMO"

    return f"""
    <style>
    @keyframes pulse-glow {{
        0%,100% {{ text-shadow: 0 0 10px rgba(251,146,60,0.6), 0 0 30px rgba(251,146,60,0.3); }}
        50%      {{ text-shadow: 0 0 20px rgba(251,146,60,1),   0 0 60px rgba(251,146,60,0.5); }}
    }}
    @keyframes slide-in {{
        from {{ opacity:0; transform:translateY(-12px); }}
        to   {{ opacity:1; transform:translateY(0); }}
    }}
    @keyframes scan-line {{
        0%   {{ transform: translateY(-100%); opacity:0.3; }}
        100% {{ transform: translateY(400%);  opacity:0; }}
    }}
    .eo-header {{
        position:relative;overflow:hidden;
        background:linear-gradient(135deg,#070f1f 0%,#0d1f3c 50%,#070f1f 100%);
        border:1px solid rgba(59,130,246,0.2);border-radius:16px;
        padding:1.5rem 2rem;margin-bottom:1.5rem;
        animation: slide-in 0.5s ease both;
    }}
    .eo-header::before {{
        content:'';position:absolute;top:0;left:0;right:0;height:1px;
        background:linear-gradient(90deg,transparent,#3b82f6,#60a5fa,transparent);
    }}
    .eo-header::after {{
        content:'';position:absolute;top:0;left:0;right:0;height:30px;
        background:linear-gradient(180deg,rgba(59,130,246,0.04),transparent);
        animation: scan-line 3s linear infinite;
        pointer-events:none;
    }}
    .eo-bolt {{
        display:inline-block;font-size:2rem;line-height:1;
        animation: pulse-glow 2s ease-in-out infinite;
    }}
    .eo-title {{
        font-family:'Space Mono',monospace;font-size:1.6rem;font-weight:700;
        color:#f1f5f9;margin:0;line-height:1.2;letter-spacing:-0.02em;
    }}
    .eo-title span {{ color:#3b82f6; }}
    .eo-subtitle {{ font-size:0.88rem;color:#475569;margin:0.2rem 0 0 0; }}
    .eo-clock {{
        font-family:'Space Mono',monospace;font-size:1rem;
        color:#3b82f6;font-weight:700;letter-spacing:0.05em;
    }}
    .eo-badge {{
        display:inline-block;font-family:'Space Mono',monospace;
        font-size:0.6rem;font-weight:700;letter-spacing:0.1em;
        text-transform:uppercase;padding:0.2rem 0.6rem;border-radius:20px;
        background:rgba(52,211,153,0.1);color:{mode_color};
        border:1px solid {mode_color}44;
    }}
    </style>

    <div class="eo-header">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:1rem;flex-wrap:wrap;">
            <div style="display:flex;align-items:center;gap:0.8rem;">
                <span class="eo-bolt">⚡</span>
                <div>
                    <p class="eo-title">Energy <span>Optimizer</span></p>
                    <p class="eo-subtitle">
                        24-hour consumption forecast &nbsp;·&nbsp;
                        <span class="eo-badge">{mode_text}</span>
                        &nbsp;{location_html}
                    </p>
                </div>
            </div>
            <div style="text-align:right;">
                <div class="eo-clock" id="live-clock">--:--:--</div>
                <div style="font-size:0.75rem;color:#1e3a5f;margin-top:0.2rem;" id="live-date">
                </div>
            </div>
        </div>
    </div>

    <script>
    (function() {{
        function updateClock() {{
            const now  = new Date();
            const time = now.toLocaleTimeString('en-GB', {{hour12:false}});
            const date = now.toLocaleDateString('en-GB', {{weekday:'long',day:'numeric',month:'long',year:'numeric'}});
            const cl   = document.getElementById('live-clock');
            const dt   = document.getElementById('live-date');
            if (cl) cl.textContent = time;
            if (dt) dt.textContent = date;
        }}
        updateClock();
        setInterval(updateClock, 1000);
    }})();
    </script>
    """


def weather_card(weather_df, location: str = None) -> str:
    """
    Compact weather card showing today's forecast conditions.
    Takes the 24-row DataFrame returned by fetch_forecast_weather().
    Shows: current hour temp, feels-like, humidity, wind, rain chance, cloud cover.
    """
    import math
    from datetime import datetime

    if weather_df is None or len(weather_df) == 0:
        return ""

    now_hour = datetime.utcnow().hour

    # Grab the row closest to current hour, fallback to row 12 (noon)
    try:
        idx = min(now_hour, len(weather_df) - 1)
        row = weather_df.iloc[idx]
    except Exception:
        row = weather_df.iloc[min(12, len(weather_df) - 1)]

    temp        = float(row.get("temperature",        0))
    feels_like  = float(row.get("apparentTemperature", temp))
    humidity    = float(row.get("humidity",           0))   # 0-1 after normalisation
    wind        = float(row.get("windSpeed",          0))
    cloud       = float(row.get("cloudCover",         0))   # 0-1
    precip_prob = float(row.get("precipProbability",  0))   # 0-1

    # Daily hi/lo
    temp_col = weather_df.get("temperature", None)
    if temp_col is not None:
        hi = float(temp_col.max())
        lo = float(temp_col.min())
    else:
        hi = lo = temp

    # Weather icon based on cloud + precip
    if precip_prob > 0.6:
        icon = "🌧"
        condition = "Rain likely"
    elif precip_prob > 0.3:
        icon = "🌦"
        condition = "Possible showers"
    elif cloud > 0.7:
        icon = "☁️"
        condition = "Overcast"
    elif cloud > 0.35:
        icon = "⛅"
        condition = "Partly cloudy"
    else:
        icon = "☀️"
        condition = "Clear"

    humidity_pct = int(humidity * 100)
    cloud_pct    = int(cloud    * 100)
    precip_pct   = int(precip_prob * 100)
    loc_label    = f"📍 {location}" if location else "Today's forecast"

    return f"""
    <style>
    *{{box-sizing:border-box;margin:0;padding:0;}}
    html,body{{background:transparent;overflow:hidden;height:auto;}}
    .wx-card{{
        display:grid;
        grid-template-columns:auto 1fr;
        gap:1.2rem;
        align-items:center;
        background:linear-gradient(135deg,#0a1628 0%,#0d1f3c 100%);
        border:1px solid rgba(59,130,246,0.2);
        border-left:3px solid #3b82f6;
        border-radius:12px;
        padding:0.9rem 1.4rem;
    }}
    .wx-icon{{font-size:2.8rem;line-height:1;}}
    .wx-main{{display:flex;flex-direction:column;gap:0.35rem;}}
    .wx-loc{{font-size:0.62rem;font-weight:700;letter-spacing:0.12em;
        text-transform:uppercase;color:#3b82f6;font-family:'Space Mono',monospace;}}
    .wx-temp{{display:flex;align-items:baseline;gap:0.5rem;}}
    .wx-temp-val{{font-family:'Space Mono',monospace;font-size:1.9rem;
        font-weight:700;color:#f1f5f9;line-height:1;}}
    .wx-temp-unit{{font-size:0.9rem;color:#64748b;}}
    .wx-cond{{font-size:0.85rem;color:#94a3b8;}}
    .wx-stats{{display:flex;gap:1.4rem;flex-wrap:wrap;margin-top:0.25rem;}}
    .wx-stat{{font-size:0.75rem;color:#64748b;white-space:nowrap;}}
    .wx-stat b{{color:#cbd5e1;font-weight:600;}}
    .wx-hilos{{font-size:0.72rem;color:#475569;}}
    .wx-hilos b{{color:#94a3b8;}}
    </style>
    <div class="wx-card">
        <div class="wx-icon">{icon}</div>
        <div class="wx-main">
            <div class="wx-loc">{loc_label}</div>
            <div class="wx-temp">
                <span class="wx-temp-val">{temp:.0f}</span>
                <span class="wx-temp-unit">°C &nbsp;·&nbsp; {condition}</span>
            </div>
            <div class="wx-stats">
                <span class="wx-stat">🌡 Feels <b>{feels_like:.0f}°C</b></span>
                <span class="wx-stat">💧 Humidity <b>{humidity_pct}%</b></span>
                <span class="wx-stat">💨 Wind <b>{wind:.0f} mph</b></span>
                <span class="wx-stat">☁️ Cloud <b>{cloud_pct}%</b></span>
                <span class="wx-stat">🌧 Rain <b>{precip_pct}%</b></span>
            </div>
            <div class="wx-hilos">Hi <b>{hi:.0f}°C</b> &nbsp;/&nbsp; Lo <b>{lo:.0f}°C</b></div>
        </div>
    </div>
    """