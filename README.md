# ⚡ Energy Optimizer

**Forecast your household electricity. Cut your bills.**

A machine learning dashboard that predicts 24-hour household energy consumption and recommends the cheapest times to run flexible appliances — washing machine, dishwasher, EV charger, and more.

**[→ Live Demo](https://energy-optimizer-97hkkpjh7nzqmswnud9ulh.streamlit.app/)** — no sign-up, works immediately in your browser.

---

## What It Does

Upload a CSV from your smart meter and the app retrains an XGBoost model on your actual usage data, then:

- Forecasts your consumption hour-by-hour for the next 24 hours
- Highlights your cheapest and most expensive demand windows
- Tells you exactly when to run each appliance to minimise cost
- Estimates how much you save vs running at peak demand
- Analyses a kWh budget and shows which appliances to cut first

No data to upload? A pre-loaded 2016 US household dataset lets you explore everything immediately in demo mode.

---

## Results

The model was trained on 349 days of hourly smart meter data and evaluated on a held-out chronological 20% test set.

| Metric | Value |
|---|---|
| Baseline MAPE (naïve lag-24h) | 52.71% |
| Model MAPE (all hours) | 30.11% |
| Model MAPE (hours > 0.3 kW) | 24.60% |
| MAE | 0.207 kW |
| Median absolute error | 0.13 kW |
| Predictions within 30% of actual | 71.3% |

> **Why the 0.3 kW filter?** Raw MAPE inflates sharply on near-zero standby hours (e.g. 135% error on a 0.05 kW reading at 3 AM). The filtered figure is the honest measure of accuracy during meaningful consumption. The model achieves a **28 percentage-point improvement** over the naive baseline.

---

## Tech Stack

| Layer | Detail |
|---|---|
| Model | XGBoost (`n_estimators=700`, `learning_rate=0.02`, `max_depth=6`) |
| Features | 31 total — time encoding, lag features (1h–168h), rolling statistics, weather |
| Dashboard | Streamlit + Plotly |
| Weather | Open-Meteo API (free, no key required) |
| Deployment | Streamlit Community Cloud |
| Tests | 16 pytest tests |

---

## How It Works

1. **Upload** — `src/ingestor.py` parses your CSV, auto-detects columns, and normalises everything to a clean hourly series
2. **Feature engineering** — `src/features.py` builds time, lag, and rolling features. The same module is used by both the training notebooks and the live dashboard — no duplicated logic
3. **Train or load** — `src/model.py` either loads the pre-trained model from `models/` or retrains XGBoost on your data with a temporal 80/20 split
4. **Weather** — `src/weather.py` optionally fetches historical and forecast weather from Open-Meteo and merges it into both training and inference
5. **Recommend** — `src/optimizer.py` turns the 24-hour forecast into appliance scheduling suggestions and budget guidance

---

## Feature Engineering

**20 core features** — always available from time and consumption alone:

- **Calendar:** hour, day of week, month, day of year, is_weekend, is_nighttime
- **Cyclical encoding:** sin/cos transforms for hour (period 24) and day-of-week (period 7), so the model understands hour 23 and hour 0 are adjacent
- **Lag features:** consumption at 1h, 2h, 3h, 24h, 48h, 168h ago — `lag_1h` is the single most important feature (importance score 0.32)
- **Rolling statistics:** 3h and 24h rolling mean, 24h rolling std and max — all computed on a shifted series to prevent lookahead

**11 optional weather features** — merged in when a location is provided: temperature, apparent temperature, humidity, wind speed, wind bearing, cloud cover, precipitation intensity, precipitation probability, pressure, dew point, visibility.

---

## Key Design Decisions

**Temporal train/test split** — strictly chronological 80/20, no shuffling. Prevents data leakage and simulates real forecasting conditions where future values are unavailable.

**No lookahead in rolling features** — rolling windows are computed on `df[TARGET].shift(1)`, so the value at position `i` only uses rows `i-1` and earlier.

**Iterative forecasting** — for multi-hour forecasts, each predicted value is fed back as the lag input for the next hour rather than using ground truth. This is how the model behaves in production.

**MAPE threshold** — accuracy is reported on hours above 0.3 kW. Raw MAPE on near-zero hours is misleading and not representative of real-world performance.

**Personalised budget planner** — the appliance impact table and cut suggestions use the user's actual selected appliances and their real wattage, not generic hardcoded defaults.

---

## Project Structure

```
energy-optimizer/
├── app.py                        # Streamlit dashboard — UI, session state, orchestration
├── src/
│   ├── features.py               # Feature engineering pipeline
│   ├── ingestor.py               # CSV parsing, normalisation, validation
│   ├── model.py                  # Model loading, retraining, iterative forecast
│   ├── optimizer.py              # Scheduling recommendations, budget planner
│   ├── weather.py                # Open-Meteo geocoding and weather fetch
│   └── ui_components.py          # Custom HTML/JS components
├── models/
│   ├── xgb_model.pkl             # Pre-trained XGBoost model
│   ├── feature_columns.json      # Feature column names
│   └── model_metadata.json       # Performance metrics
├── data/processed/
│   ├── hourly_data.csv           # Cleaned hourly smart meter data
│   └── feature_matrix.csv        # Pre-built feature matrix
├── tests/
│   └── test_pipeline.py          # 16 pytest tests
└── notebooks/
    ├── 01_eda.ipynb
    ├── 02_feature_engineering.ipynb
    └── 03_model_training.ipynb
```

---

## Running Locally

```bash
git clone https://github.com/Sam4larin/Energy-Optimizer.git
cd Energy-Optimizer

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\Activate.ps1       # Windows PowerShell

pip install --upgrade pip
pip install -r requirements.txt

streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Tests

```bash
pip install pytest
pytest tests/ -v
```

16 tests covering feature shape and nulls, cyclical feature bounds, lag correctness, forecast output dimensions, savings non-negativity, staggered appliance windows, and budget arithmetic.

---

## Supported CSV Formats

The ingestor is built to handle real-world meter exports without manual cleaning:

- Comma, semicolon, and tab-separated files
- Unix timestamps in seconds or milliseconds
- Separate Date and Time columns
- Sub-hourly data (e.g. 30-minute intervals) — resampled to hourly
- Duplicate timestamps — averaged
- Missing hourly gaps — filled

**Minimum requirement:** one timestamp column, one numeric consumption column, at least 14 days of readings.

---

## Dataset

**Smart Home Dataset with Weather Information** (Kaggle). One US household, January–December 2016, recorded at 1-minute intervals, resampled to hourly. 8,399 hourly observations after cleaning, zero nulls, continuous 1-hour intervals throughout.

---

## Troubleshooting

**Upload validation fails**
- Ensure the file has at least 14 days of readings
- Check the timestamp column contains real dates or Unix timestamps
- Check the consumption column is numeric whole-home usage, not a sub-meter

**Forecast looks off for uploaded data**
- Verify timestamps are in the correct timezone and chronological order
- Make sure the consumption column is total household load, not a single circuit
- More history (30+ days) produces more stable training

**Weather lookup fails**
- Try a more specific name: `Lagos, NG` or `London, UK`
- The app continues without weather — forecasts still work using time and lag features only