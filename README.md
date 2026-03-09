# ⚡ Energy Optimizer

A machine learning dashboard that forecasts household energy consumption 24 hours ahead and recommends the cheapest times to run flexible appliances.

**[Live Demo →](https://energy-optimizer-97hkkpjh7nzqmswnud9ulh.streamlit.app/)**

---

## What It Does

- Forecasts 24-hour energy consumption using XGBoost trained on 349 days of smart meter data
- Identifies peak and trough demand windows
- Recommends optimal run times for dishwasher, washing machine, tumble dryer, and EV charger
- Estimates cost savings vs running appliances at peak demand

## Results

| Metric | Value |
|---|---|
| Baseline MAPE (naïve lag-24h) | 52.71% |
| Model MAPE (all hours) | 30.11% |
| Model MAPE (hours > 0.3 kW) | 24.60% |
| MAE | 0.207 kW |
| Median absolute error | 0.13 kW |

28 percentage point improvement over baseline. 71.3% of predictions within 30% of actual on meaningful consumption hours.

## Tech Stack

- **Model:** XGBoost (n_estimators=700, learning_rate=0.02, max_depth=6)
- **Features:** 31 features — time cyclical encoding, lag features (1h to 168h), rolling statistics, weather variables
- **Dashboard:** Streamlit + Plotly
- **Deployment:** Streamlit Community Cloud

## Project Structure
```
energy-optimizer/
├── app.py                  # Streamlit dashboard
├── src/
│   ├── features.py         # Feature engineering pipeline
│   ├── model.py            # Model loading and forecast generation
│   └── optimizer.py        # Appliance scheduling recommendations
├── models/
│   ├── xgb_model.pkl       # Trained XGBoost model
│   ├── feature_columns.json
│   └── model_metadata.json
├── data/processed/
│   ├── hourly_data.csv     # Cleaned hourly smart meter data
│   └── feature_matrix.csv  # Engineered feature matrix
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
└── tests/
    └── test_pipeline.py    # 16 pytest tests
```

## Dataset

Smart Home Dataset with Weather Information (Kaggle). One US household, January–December 2016, recorded at 1-minute intervals, resampled to hourly. 8,399 hourly observations after cleaning.

## Key Design Decisions

**Temporal train/test split:** Chronological 80/20 split (no random shuffling) to prevent data leakage and simulate real forecasting conditions.

**Lag features as primary signal:** `lag_1h` is the most important feature (importance score 0.32). The model learns that recent consumption is the strongest predictor of near-future consumption.

**Iterative forecasting:** For multi-hour forecasts, predictions are fed back as lag inputs rather than using ground truth — this is how the model would work in production.

**MAPE threshold:** Raw MAPE is misleading on near-zero hours (e.g. 135% error on 0.19 kW actual). Reported MAPE filters to hours above 0.3 kW for an honest evaluation.

## Running Locally
```bash
git clone https://github.com/Sam4larin/Energy-Optimizer.git
cd Energy-Optimizer
pip install -r requirements.txt
streamlit run app.py
```

## Tests
```bash
pytest tests/ -v
```

16 tests covering feature engineering, model loading, forecast generation, and optimizer logic.