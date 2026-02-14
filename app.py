import time
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from forecasting.metrics import mape, rmse, mae, bias
from forecasting.classical import (
    moving_average_forecast,
    exp_smoothing_forecast,
    holt_winters_forecast,
)
from forecasting.prophet_model import prophet_forecast
from forecasting.ml_xgb import xgboost_forecast

st.set_page_config(page_title="Demand Forecast Visualizer", layout="wide")

# ---------------- Helpers ----------------
def load_sample_weekly():
    return pd.read_csv("data/sample_weekly_retail.csv")

def validate_and_prepare(df: pd.DataFrame, date_col: str, y_col: str) -> pd.Series:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col, y_col])
    out = out.sort_values(date_col).set_index(date_col)

    out[y_col] = pd.to_numeric(out[y_col], errors="coerce")
    out = out.dropna(subset=[y_col])

    # infer/regularize frequency
    freq = pd.infer_freq(out.index)
    if freq is None:
        # fallback: daily resample
        out = out.resample("D").sum()

    out = out.asfreq(pd.infer_freq(out.index) or "D")
    out[y_col] = out[y_col].interpolate(limit_direction="both")

    return out[y_col].rename("demand")

@st.cache_data(show_spinner=False)
def run_forecasts(y: pd.Series, horizon: int, train_ratio: float, config: dict):
    n = len(y)
    split = int(n * train_ratio)
    y_train = y.iloc[:split]
    y_test = y.iloc[split : split + horizon]

    results = {}
    timings = {}

    def _timeit(name, fn):
        t0 = time.time()
        fc = fn()
        timings[name] = time.time() - t0
        results[name] = fc

    if config["use_ma"]:
        _timeit(
            "Moving Average",
            lambda: moving_average_forecast(
                y_train, horizon, window=config["ma_window"]
            ),
        )

    if config["use_es"]:
        _timeit(
            "Exponential Smoothing",
            lambda: exp_smoothing_forecast(
                y_train, horizon, alpha=config["es_alpha"]
            ),
        )

    if config["use_hw"]:
        _timeit(
            "Holt-Winters",
            lambda: holt_winters_forecast(
                y_train,
                horizon,
                seasonal_periods=config["hw_seasonal_periods"],
                trend=config["hw_trend"],
                seasonal=config["hw_seasonal"],
            ),
        )

    if config["use_prophet"]:
        _timeit(
            "Prophet",
            lambda: prophet_forecast(
                y_train, horizon, seasonality_mode=config["prophet_mode"]
            ),
        )

    if config["use_xgb"]:
        _timeit(
            "XGBoost",
            lambda: xgboost_forecast(
                y_train,
                horizon,
                n_estimators=config["xgb_estimators"],
                max_depth=config["xgb_depth"],
                learning_rate=config["xgb_lr"],
            ),
        )

    metrics_rows = []
    for name, fc in results.items():
        common_idx = fc.index.intersection(y_test.index)
        if len(common_idx) > 0:
            yt = y_test.loc[common_idx]
            yp = fc.loc[common_idx]
            metrics_rows.append(
                {
                    "Algorithm": name,
                    "MAPE (%)": mape(yt, yp),
                    "RMSE": rmse(yt, yp),
                    "MAE": mae(yt, yp),
                    "Bias": bias(yt, yp),
                    "Train Time (s)": timings.get(name, np.nan),
                }
            )
        else:
            metrics_rows.append(
                {
                    "Algorithm": name,
                    "MAPE (%)": np.nan,
                    "RMSE": np.nan,
                    "MAE": np.nan,
                    "Bias": np.nan,
                    "Train Time (s)": timings.get(name, np.nan),
                }
            )

    metrics_df = (
        pd.DataFrame(metrics_rows)
        .sort_values(by="MAPE (%)", na_position="last")
        .reset_index(drop=True)
    )
    return y_train, y_test, results, metrics_df

def make_forecast_chart(y_train, y_test, forecasts: dict):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_train.index,
            y=y_train.values,
            mode="lines",
            name="Train (Actual)",
        )
    )
    if len(y_test) > 0:
        fig.add_trace(
            go.Scatter(
                x=y_test.index,
                y=y_test.values,
                mode="lines",
                name="Test (Actual)",
            )
        )

    for name, fc in forecasts.items():
        fig.add_trace(
            go.Scatter(
                x=fc.index,
                y=fc.values,
                mode="lines",
                name=name,
                line=dict(dash="dash"),
            )
        )

    fig.update_layout(
        title="Demand Forecast Comparison",
        xaxis_title="Date",
        yaxis_title="Demand",
        hovermode="x unified",
        legend_title="Series",
    )
    return fig

# ---------------- UI ----------------
st.title("📈 Demand Forecast Visualizer")
st.caption("Compare classical + AI/ML forecasting methods on the same demand dataset.")

with st.sidebar:
    st.header("Data Input")
    data_mode = st.radio(
        "Choose data source", ["Upload CSV", "Use sample dataset"], index=1
    )

    if data_mode == "Upload CSV":
        file = st.file_uploader("Upload CSV", type=["csv"])
        df_raw = pd.read_csv(file) if file else None
    else:
        df_raw = load_sample_weekly()

    st.divider()
    st.header("Columns")
    date_col = st.text_input("Date column", value="date")
    y_col = st.text_input("Demand column", value="demand")

    st.divider()
    st.header("Forecast Settings")
    horizon = st.number_input(
        "Forecast horizon (periods)", min_value=1, max_value=200, value=12, step=1
    )
    train_ratio = st.slider(
        "Train/Test split", min_value=0.5, max_value=0.95, value=0.8, step=0.05
    )

    st.divider()
    st.header("Algorithms")

    use_ma = st.checkbox("Moving Average", value=True)
    ma_window = st.selectbox("MA window", [3, 7, 12, 26], index=1, disabled=not use_ma)

    use_es = st.checkbox("Exponential Smoothing", value=True)
    es_alpha = st.slider(
        "ES alpha",
        min_value=0.01,
        max_value=0.99,
        value=0.3,
        step=0.01,
        disabled=not use_es,
    )

    use_hw = st.checkbox("Holt-Winters", value=False)
    hw_seasonal_periods = st.selectbox(
        "HW seasonal periods",
        [4, 7, 12, 26, 52],
        index=2,
        disabled=not use_hw,
    )
    hw_trend = st.selectbox("HW trend", ["add", "mul", "None"], index=0, disabled=not use_hw)
    hw_seasonal = st.selectbox("HW seasonal", ["add", "mul"], index=1, disabled=not use_hw)

    use_prophet = st.checkbox("Prophet", value=True)
    prophet_mode = st.selectbox(
        "Prophet seasonality mode",
        ["multiplicative", "additive"],
        index=0,
        disabled=not use_prophet,
    )

    use_xgb = st.checkbox("XGBoost", value=True)
    xgb_estimators = st.slider("XGB estimators", 50, 1000, 300, 50, disabled=not use_xgb)
    xgb_depth = st.slider("XGB max_depth", 2, 12, 5, 1, disabled=not use_xgb)
    xgb_lr = st.slider("XGB learning_rate", 0.01, 0.3, 0.05, 0.01, disabled=not use_xgb)

    run_btn = st.button("Run Forecasts", type="primary")

if df_raw is None:
    st.info("Upload a CSV to begin.")
    st.stop()

try:
    y = validate_and_prepare(df_raw, date_col=date_col, y_col=y_col)
except Exception as e:
    st.error(f"Data validation failed: {e}")
    st.stop()

st.subheader("Data Preview")
st.dataframe(pd.DataFrame({"demand": y}).head(20), use_container_width=True)

if run_btn:
    cfg = dict(
        use_ma=use_ma,
        ma_window=int(ma_window),
        use_es=use_es,
        es_alpha=float(es_alpha),
        use_hw=use_hw,
        hw_seasonal_periods=int(hw_seasonal_periods),
        hw_trend=(None if hw_trend == "None" else hw_trend),
        hw_seasonal=str(hw_seasonal),
        use_prophet=use_prophet,
        prophet_mode=str(prophet_mode),
        use_xgb=use_xgb,
        xgb_estimators=int(xgb_estimators),
        xgb_depth=int(xgb_depth),
        xgb_lr=float(xgb_lr),
    )

    with st.spinner("Training models and generating forecasts..."):
        y_train, y_test, forecasts, metrics_df = run_forecasts(
            y, horizon=int(horizon), train_ratio=float(train_ratio), config=cfg
        )

    st.subheader("Forecast Chart")
    st.plotly_chart(make_forecast_chart(y_train, y_test, forecasts), use_container_width=True)

    st.subheader("Accuracy Metrics")
    st.dataframe(metrics_df, use_container_width=True)

    st.subheader("Export")
    export_df = pd.DataFrame({"actual": y_test})
    for name, fc in forecasts.items():
        export_df[name] = fc.reindex(export_df.index)

    csv = export_df.to_csv(index=True).encode("utf-8")
    st.download_button(
        "Download Forecasts CSV",
        data=csv,
        file_name="forecasts.csv",
        mime="text/csv",
    )
else:
    st.info("Adjust settings in the sidebar and click **Run Forecasts**.")
