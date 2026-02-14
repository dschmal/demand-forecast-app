"""
Demand Forecast Visualizer
Built for simple Streamlit deployment (single-file app; no local package imports).
"""

import time
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Optional dependencies (app still runs if these fail to import)
try:
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
    _HAS_STATSMODELS = True
except Exception:
    _HAS_STATSMODELS = False

try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

try:
    from prophet import Prophet
    _HAS_PROPHET = True
except Exception:
    _HAS_PROPHET = False


# ---------------- Page config ----------------
st.set_page_config(
    page_title="Demand Forecast Visualizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- Dark theme CSS (similar style to your EOQ app) ----------------
st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1, h2, h3 { color: #FFFFFF !important; font-weight: 600 !important; }
    h1 { font-size: 28px !important; margin-bottom: 0.5rem !important; }
    h2 { font-size: 18px !important; margin-top: 2rem !important; margin-bottom: 1.2rem !important; }

    .subtitle { color: #A0AEC0; font-size: 14px; margin-bottom: 1.8rem; }

    [data-testid="stMetric"] {
        background-color: #1A202C;
        padding: 1.2rem;
        border-radius: 8px;
        border: 1px solid #2D3748;
    }
    [data-testid="stMetricValue"] { color: #FFFFFF !important; font-size: 26px !important; font-weight: 700 !important; }
    [data-testid="stMetricLabel"] { color: #E2E8F0 !important; font-size: 12px !important; font-weight: 600 !important; text-transform: uppercase; }

    section[data-testid="stSidebar"] { background-color: #1A202C; }
    section[data-testid="stSidebar"] > div { background-color: #1A202C; }
    section[data-testid="stSidebar"] h2 {
        color: #FFFFFF !important; font-size: 16px !important;
        margin-bottom: 1.2rem !important; padding-bottom: 0.5rem !important;
        border-bottom: 1px solid #2D3748 !important;
    }
    section[data-testid="stSidebar"] label { color: #E2E8F0 !important; font-weight: 500 !important; font-size: 13px !important; }
    section[data-testid="stSidebar"] input { background-color: #2D3748 !important; color: #FFFFFF !important; border: 1px solid #4A5568 !important; }

    .metric-pill {
        display: inline-block;
        background-color: #2B6CB0;
        color: white;
        padding: 0.35rem 0.8rem;
        border-radius: 16px;
        font-size: 12px;
        font-weight: 700;
        margin-right: 0.4rem;
        margin-bottom: 0.3rem;
    }
    .info-pill {
        display: inline-block;
        background-color: #2D3748;
        color: #A0AEC0;
        padding: 0.35rem 0.7rem;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-bottom: 0.3rem;
    }

    .stButton > button {
        background-color: #3182CE !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 700 !important;
    }
    .stButton > button:hover { background-color: #2C5282 !important; }

    .stDownloadButton > button {
        background-color: #38A169 !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 700 !important;
    }
    .stDownloadButton > button:hover { background-color: #2F855A !important; }

    .streamlit-expanderHeader {
        background-color: #1A202C !important;
        color: #E2E8F0 !important;
        border: 1px solid #2D3748 !important;
    }
    </style>
""", unsafe_allow_html=True)


# ---------------- Metrics ----------------
def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mape(y_true, y_pred, eps=1e-9):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

def bias(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(y_pred - y_true))


# ---------------- Forecasting methods ----------------
def _infer_freq(idx: pd.DatetimeIndex) -> str:
    return pd.infer_freq(idx) or "D"

def moving_average_forecast(y: pd.Series, horizon: int, window: int = 7) -> pd.Series:
    last = float(y.rolling(window=window).mean().iloc[-1])
    idx = pd.date_range(start=y.index[-1], periods=horizon + 1, freq=_infer_freq(y.index))[1:]
    return pd.Series([last] * horizon, index=idx, name="Moving Average")

def exp_smoothing_forecast(y: pd.Series, horizon: int, alpha: float = 0.3) -> pd.Series:
    if not _HAS_STATSMODELS:
        raise RuntimeError("statsmodels is not available in this environment.")
    model = SimpleExpSmoothing(y, initialization_method="estimated").fit(
        smoothing_level=alpha, optimized=False
    )
    fc = model.forecast(horizon)
    fc.name = "Exponential Smoothing"
    return fc

def holt_winters_forecast(y: pd.Series, horizon: int, seasonal_periods: int, trend: str | None, seasonal: str) -> pd.Series:
    if not _HAS_STATSMODELS:
        raise RuntimeError("statsmodels is not available in this environment.")
    model = ExponentialSmoothing(
        y,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods,
        initialization_method="estimated",
    ).fit(optimized=True)
    fc = model.forecast(horizon)
    fc.name = "Holt-Winters"
    return fc

def prophet_forecast(y: pd.Series, horizon: int, seasonality_mode: str) -> pd.Series:
    if not _HAS_PROPHET:
        raise RuntimeError("prophet is not available in this environment.")
    df = pd.DataFrame({"ds": y.index, "y": y.values})
    m = Prophet(
        seasonality_mode=seasonality_mode,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
    )
    m.fit(df)
    future = m.make_future_dataframe(periods=horizon, freq=_infer_freq(y.index))
    fc = m.predict(future)[["ds", "yhat"]].set_index("ds")["yhat"].iloc[-horizon:]
    fc.name = "Prophet"
    return fc

def _make_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["month"] = out.index.month
    out["quarter"] = out.index.quarter
    out["day_of_week"] = out.index.dayofweek
    out["week_of_year"] = out.index.isocalendar().week.astype(int)

    out["lag_1"] = out["y"].shift(1)
    out["lag_7"] = out["y"].shift(7)
    out["lag_14"] = out["y"].shift(14)

    out["roll_mean_4"] = out["y"].rolling(4).mean()
    out["roll_std_4"] = out["y"].rolling(4).std()
    return out

def xgboost_forecast(y: pd.Series, horizon: int, n_estimators: int, max_depth: int, learning_rate: float) -> pd.Series:
    if not _HAS_XGB:
        raise RuntimeError("xgboost is not available in this environment.")
    df = pd.DataFrame({"y": y})
    feat = _make_time_features(df).dropna()
    X = feat.drop(columns=["y"])
    y_train = feat["y"]

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective="reg:squarederror",
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    model.fit(X, y_train)

    idx_future = pd.date_range(start=y.index[-1], periods=horizon + 1, freq=_infer_freq(y.index))[1:]
    history = df.copy()
    preds = []
    for ts in idx_future:
        tmp = pd.DataFrame({"y": history["y"]})
        tmp.index = history.index
        x_last = _make_time_features(tmp).iloc[[-1]].drop(columns=["y"])
        yhat = float(model.predict(x_last)[0])
        preds.append(yhat)
        history.loc[ts, "y"] = yhat

    return pd.Series(preds, index=idx_future, name="XGBoost")


# ---------------- Data utilities ----------------
def load_sample():
    return pd.read_csv("data/sample_weekly_retail.csv")

def validate_and_prepare(df: pd.DataFrame, date_col: str, y_col: str) -> pd.Series:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col, y_col])
    out = out.sort_values(date_col).set_index(date_col)

    out[y_col] = pd.to_numeric(out[y_col], errors="coerce")
    out = out.dropna(subset=[y_col])

    # if frequency can't be inferred, resample to daily sum (safe default)
    if pd.infer_freq(out.index) is None:
        out = out.resample("D").sum()

    out = out.asfreq(_infer_freq(out.index))
    out[y_col] = out[y_col].interpolate(limit_direction="both")
    return out[y_col].rename("demand")


@st.cache_data(show_spinner=False)
def run_all(y: pd.Series, horizon: int, train_ratio: float, cfg: dict):
    n = len(y)
    split = max(2, int(n * train_ratio))
    y_train = y.iloc[:split]
    y_test = y.iloc[split : split + horizon]

    forecasts = {}
    timings = {}
    errors = {}

    def _timeit(name, fn):
        t0 = time.time()
        try:
            fc = fn()
            forecasts[name] = fc
        except Exception as e:
            errors[name] = str(e)
        timings[name] = time.time() - t0

    if cfg["use_ma"]:
        _timeit("Moving Average", lambda: moving_average_forecast(y_train, horizon, cfg["ma_window"]))

    if cfg["use_es"]:
        _timeit("Exponential Smoothing", lambda: exp_smoothing_forecast(y_train, horizon, cfg["es_alpha"]))

    if cfg["use_hw"]:
        _timeit("Holt-Winters", lambda: holt_winters_forecast(
            y_train, horizon,
            seasonal_periods=cfg["hw_seasonal_periods"],
            trend=cfg["hw_trend"],
            seasonal=cfg["hw_seasonal"],
        ))

    if cfg["use_prophet"]:
        _timeit("Prophet", lambda: prophet_forecast(y_train, horizon, cfg["prophet_mode"]))

    if cfg["use_xgb"]:
        _timeit("XGBoost", lambda: xgboost_forecast(
            y_train, horizon,
            n_estimators=cfg["xgb_estimators"],
            max_depth=cfg["xgb_depth"],
            learning_rate=cfg["xgb_lr"],
        ))

    rows = []
    for name, fc in forecasts.items():
        common = fc.index.intersection(y_test.index)
        if len(common) > 0:
            yt = y_test.loc[common]
            yp = fc.loc[common]
            rows.append({
                "Algorithm": name,
                "MAPE (%)": mape(yt, yp),
                "RMSE": rmse(yt, yp),
                "MAE": mae(yt, yp),
                "Bias": bias(yt, yp),
                "Train Time (s)": timings.get(name, np.nan),
            })
        else:
            rows.append({
                "Algorithm": name,
                "MAPE (%)": np.nan,
                "RMSE": np.nan,
                "MAE": np.nan,
                "Bias": np.nan,
                "Train Time (s)": timings.get(name, np.nan),
            })

    metrics_df = pd.DataFrame(rows).sort_values("MAPE (%)", na_position="last").reset_index(drop=True)
    return y_train, y_test, forecasts, metrics_df, errors


def make_chart(y_train, y_test, forecasts: dict):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_train.index, y=y_train.values, mode="lines", name="Train (Actual)"))
    if len(y_test) > 0:
        fig.add_trace(go.Scatter(x=y_test.index, y=y_test.values, mode="lines", name="Test (Actual)"))
    for name, fc in forecasts.items():
        fig.add_trace(go.Scatter(x=fc.index, y=fc.values, mode="lines", name=name, line=dict(dash="dash")))

    fig.update_layout(
        height=440,
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        font=dict(size=12, color="#E2E8F0"),
        xaxis=dict(showgrid=True, gridcolor="#2D3748", gridwidth=0.5, title="Date", title_font=dict(color="#A0AEC0")),
        yaxis=dict(showgrid=True, gridcolor="#2D3748", gridwidth=0.5, title="Demand", title_font=dict(color="#A0AEC0")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#E2E8F0")),
        margin=dict(l=60, r=40, t=60, b=60),
        hovermode="x unified",
    )
    return fig


# ---------------- Header ----------------
st.markdown("# 📈 Demand Forecast Visualizer")
st.markdown("<p class='subtitle'>Compare forecasting methods on your demand history (deploy-safe single-file app)</p>", unsafe_allow_html=True)

st.markdown("""
<div style='margin-bottom: 1.6rem;'>
    <span class='metric-pill'>Moving Average</span>
    <span class='metric-pill'>Exponential Smoothing</span>
    <span class='metric-pill'>Holt-Winters</span>
    <span class='metric-pill'>Prophet</span>
    <span class='metric-pill'>XGBoost</span>
</div>
""", unsafe_allow_html=True)


# ---------------- Sidebar ----------------
st.sidebar.markdown("## ⚙️ Data & Settings")

data_mode = st.sidebar.radio("Data source", ["Upload CSV", "Use sample dataset"], index=1)
if data_mode == "Upload CSV":
    file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    df_raw = pd.read_csv(file) if file else None
else:
    df_raw = load_sample()

st.sidebar.markdown("### 🧾 Columns")
date_col = st.sidebar.text_input("Date column", value="date")
y_col = st.sidebar.text_input("Demand column", value="demand")

st.sidebar.markdown("### 🔮 Forecast settings")
horizon = st.sidebar.number_input("Forecast horizon (periods)", min_value=1, max_value=365, value=12, step=1)
train_ratio = st.sidebar.slider("Train/Test split", min_value=0.5, max_value=0.95, value=0.8, step=0.05)

st.sidebar.markdown("### 🧠 Algorithms")
use_ma = st.sidebar.checkbox("Moving Average", value=True)
ma_window = st.sidebar.selectbox("MA window", [3, 7, 12, 26], index=1, disabled=not use_ma)

use_es = st.sidebar.checkbox("Exponential Smoothing", value=True, disabled=not _HAS_STATSMODELS)
es_alpha = st.sidebar.slider("ES alpha", 0.01, 0.99, 0.3, 0.01, disabled=not (use_es and _HAS_STATSMODELS))

use_hw = st.sidebar.checkbox("Holt-Winters", value=False, disabled=not _HAS_STATSMODELS)
hw_seasonal_periods = st.sidebar.selectbox("HW seasonal periods", [4, 7, 12, 26, 52], index=2, disabled=not (use_hw and _HAS_STATSMODELS))
hw_trend = st.sidebar.selectbox("HW trend", ["add", "mul", "None"], index=0, disabled=not (use_hw and _HAS_STATSMODELS))
hw_seasonal = st.sidebar.selectbox("HW seasonal", ["add", "mul"], index=1, disabled=not (use_hw and _HAS_STATSMODELS))

use_prophet = st.sidebar.checkbox("Prophet", value=False, disabled=not _HAS_PROPHET)
prophet_mode = st.sidebar.selectbox("Prophet seasonality mode", ["multiplicative", "additive"], index=0, disabled=not (use_prophet and _HAS_PROPHET))

use_xgb = st.sidebar.checkbox("XGBoost", value=True, disabled=not _HAS_XGB)
xgb_estimators = st.sidebar.slider("XGB estimators", 50, 800, 300, 50, disabled=not (use_xgb and _HAS_XGB))
xgb_depth = st.sidebar.slider("XGB max_depth", 2, 12, 5, 1, disabled=not (use_xgb and _HAS_XGB))
xgb_lr = st.sidebar.slider("XGB learning_rate", 0.01, 0.3, 0.05, 0.01, disabled=not (use_xgb and _HAS_XGB))

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='color: #A0AEC0; font-size: 12px; padding: 0.5rem 0 0.5rem 0;'>
    <strong style='color: #E2E8F0;'>Demand Forecast App</strong><br>
    Single-file Streamlit app for easy deployment
</div>
""", unsafe_allow_html=True)

run_btn = st.sidebar.button("Run Forecasts", type="primary")


# ---------------- Main ----------------
if df_raw is None:
    st.info("Upload a CSV to begin.")
    st.stop()

try:
    y = validate_and_prepare(df_raw, date_col=date_col, y_col=y_col)
except Exception as e:
    st.error(f"Data validation failed: {e}")
    st.stop()

st.markdown("## 🧾 Data preview")
st.dataframe(pd.DataFrame({"demand": y}).head(24), use_container_width=True)

# Capability pills
caps = []
caps.append(f"Freq: {_infer_freq(y.index)}")
caps.append(f"Rows: {len(y):,}")
if not _HAS_PROPHET:
    caps.append("Prophet: not installed")
if not _HAS_XGB:
    caps.append("XGBoost: not installed")
if not _HAS_STATSMODELS:
    caps.append("statsmodels: not installed")

st.markdown("<div style='margin-top:0.2rem; margin-bottom:1.4rem;'>" + "".join([f"<span class='info-pill'>{c}</span>" for c in caps]) + "</div>", unsafe_allow_html=True)

if run_btn:
    cfg = dict(
        use_ma=use_ma, ma_window=int(ma_window),
        use_es=bool(use_es) and _HAS_STATSMODELS, es_alpha=float(es_alpha),
        use_hw=bool(use_hw) and _HAS_STATSMODELS, hw_seasonal_periods=int(hw_seasonal_periods),
        hw_trend=(None if hw_trend == "None" else hw_trend),
        hw_seasonal=str(hw_seasonal),
        use_prophet=bool(use_prophet) and _HAS_PROPHET, prophet_mode=str(prophet_mode),
        use_xgb=bool(use_xgb) and _HAS_XGB, xgb_estimators=int(xgb_estimators), xgb_depth=int(xgb_depth), xgb_lr=float(xgb_lr),
    )

    with st.spinner("Training models and generating forecasts..."):
        y_train, y_test, forecasts, metrics_df, errors = run_all(y, int(horizon), float(train_ratio), cfg)

    st.markdown("## 📌 Key metrics")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Train rows", f"{len(y_train):,}")
    with c2:
        st.metric("Test rows", f"{len(y_test):,}")
    with c3:
        st.metric("Horizon", f"{int(horizon)}")
    with c4:
        st.metric("Models run", f"{len(forecasts):,}")

    if len(errors) > 0:
        with st.expander("⚠️ Some models failed to run (click to view)"):
            for k, v in errors.items():
                st.write(f"**{k}**: {v}")

    st.markdown("## 📊 Forecast comparison")
    st.plotly_chart(make_chart(y_train, y_test, forecasts), use_container_width=True, config={"displayModeBar": False})

    st.markdown("## ✅ Accuracy metrics")
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    st.markdown("## 💾 Export")
    export_df = pd.DataFrame({"actual": y_test})
    for name, fc in forecasts.items():
        export_df[name] = fc.reindex(export_df.index)

    csv = export_df.to_csv(index=True).encode("utf-8")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button("📥 Download Forecasts CSV", data=csv, file_name=f"forecasts_{ts}.csv", mime="text/csv")

    with st.expander("🔧 Technical details"):
        st.markdown("""
        **What this app does**
        - Validates + cleans a demand time series (date + demand column)
        - Splits into train/test
        - Runs selected models and compares forecasts
        - Calculates MAPE / RMSE / MAE / Bias
        - Lets you download a CSV of the forecast window

        **Dependencies**
        - Always: streamlit, pandas, numpy, plotly
        - Optional: statsmodels (ES + Holt-Winters), xgboost (XGBoost), prophet (Prophet)

        If an optional library is not available in your deployment environment,
        the corresponding checkbox is disabled automatically.
        """)
else:
    st.info("Adjust settings in the sidebar and click **Run Forecasts**.")
