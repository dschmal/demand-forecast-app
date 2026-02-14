# Demand Forecast Streamlit App

A deployable Streamlit application to compare multiple demand forecasting approaches:
- Moving Average
- Exponential Smoothing
- Holt-Winters (ETS)
- Prophet
- XGBoost (time features + recursive forecast)

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Use your own data
Upload a CSV with:
- a date column (default: `date`)
- a demand/target column (default: `demand`)

Then select algorithms + parameters in the sidebar and click **Run Forecasts**.
