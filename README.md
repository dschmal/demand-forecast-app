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

## Deploy (Streamlit Community Cloud)
1. Push this repo to GitHub
2. In Streamlit Cloud, create a new app
3. Select this repo/branch
4. Set the main file to `app.py`
5. Deploy

## Docker
```bash
docker build -t demand-forecast .
docker run -p 8501:8501 demand-forecast
```
