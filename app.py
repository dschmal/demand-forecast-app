"""
Demand Forecast Visualizer (Streamlit)
Optimized for modeling + forecasting, single-file deployment.

Bucket selection: Day / Week / Month (resampling based on imported data).
Keeps Plotly visualization and adds stronger baselines + safer ETS/HW + improved XGBoost features.
"""

import time
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Optional dependencies (app still runs if these fail to import)
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
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

# ---------------- Dark theme CSS ----------------
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


# ---------------- Bucket utilities ----------------
BUCKET_TO_RULE = {
    "Day": "D",
    "Week": "W-MON",  # weekly buckets starting Monday (nice for business data)
    "Month": "MS",    # month start
}

def _suggest_bucket_from_dates(dt_index: pd.DatetimeIndex) -> str:
    if len(dt_index) < 3:
        return "Day"
    diffs = dt_index.sort_values().to_series().diff().dropna()
    if diffs.empty:
        return "Day"
    median_days = diffs.median() / pd.Timedelta(days=1)
    if median_days <= 2:
        return "Day"
    if median_days <= 14:
        return "Week"
    return "Month"

def _seasonal_period_for_bucket(bucket: str) -> int:
    # sensible defaults for demand
    if bucket == "Day":
        return 7
    if bucket == "Week":
        return 52
    return 12

def _future_index(last_ts: pd.Timestamp, periods: int, rule: str) -> pd.DatetimeIndex:
    # start AFTER the last observed bucket
    rng = pd.date_range(start=last_ts, periods=periods + 1, freq=rule)
    return rng[1:]


# ---------------- Forecasting methods ----------------
def naive_forecast(y: pd.Series, horizon: int, rule: str) -> pd.Series:
    idx = _future_index(y.index[-1], horizon, rule)
    fc = pd.Series([float(y.iloc[-1])] * horizon, index=idx, name="Naive")
    return fc

def seasonal_naive_forecast(y: pd.Series, horizon: int, rule: str, seasonal_periods: int) -> pd.Series:
    idx = _future_index(y.index[-1], horizon, rule)
    if seasonal_periods <= 1 or len(y) < seasonal_periods:
        # fallback to naive
        return naive_forecast(y, horizon, rule).rename("Seasonal Naive")
    season = y.iloc[-seasonal_periods:].values
    reps = int(np.ceil(horizon / seasonal_periods))
    preds = np.tile(season, reps)[:horizon]
    return pd.Series(preds, index=idx, name="Seasonal Naive")

def moving_average_level_forecast(y: pd.Series, horizon: int, rule: str, window: int = 7) -> pd.Series:
    window = max(1, int(window))
    level = float(y.iloc[-window:].mean()) if len(y) >= window else float(y.mean())
    idx = _future_index(y.index[-1], horizon, rule)
    return pd.Series([level] * horizon, index=idx, name="Moving Average (level)")

def ets_forecast(y: pd.Series, horizon: int, rule: str, seasonal_periods: int, seasonal_mode: str, trend_mode: str) -> pd.Series:
    """
    ETS via statsmodels ExponentialSmoothing.
    Safety:
      - Only enable seasonality if enough history exists.
      - If multiplicative is chosen, require strictly positive series.
    """
    if not _HAS_STATSMODELS:
        raise RuntimeError("statsmodels is not available in this environment.")

    y_in = y.astype(float).copy()

    use_seasonal = seasonal_periods >= 2 and len(y_in) >= (2 * seasonal_periods + 4)
    seasonal = seasonal_mode if use_seasonal else None

    if (seasonal == "mul" or trend_mode == "mul") and (y_in <= 0).any():
        raise RuntimeError("Multiplicative ETS requires all demand values > 0.")

    model = ExponentialSmoothing(
        y_in,
        trend=trend_mode,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods if use_seasonal else None,
        initialization_method="estimated",
    ).fit(optimized=True)

    fc = model.forecast(horizon)
    fc.index = _future_index(y.index[-1], horizon, rule)
    fc.name = f"ETS ({trend_mode or 'None'}/{seasonal_mode if use_seasonal else 'None'})"
    return fc

def prophet_forecast(y: pd.Series, horizon: int, rule: str, seasonality_mode: str) -> pd.Series:
    if not _HAS_PROPHET:
        raise RuntimeError("prophet is not available in this environment.")
    df = pd.DataFrame({"ds": y.index, "y": y.values})
    # Let Prophet infer seasonalities; add common ones:
    m = Prophet(seasonality_mode=seasonality_mode)
    if rule == "D":
        m.add_seasonality(name="weekly", period=7, fourier_order=6)
        m.add_seasonality(name="yearly", period=365.25, fourier_order=10)
    elif rule.startswith("W"):
        m.add_seasonality(name="yearly", period=52, fourier_order=10)
    else:
        m.add_seasonality(name="yearly", period=12, fourier_order=8)

    m.fit(df)
    future = m.make_future_dataframe(periods=horizon, freq=rule)
    pred = m.predict(future)[["ds", "yhat"]].set_index("ds")["yhat"].iloc[-horizon:]
    pred.index = _future_index(y.index[-1], horizon, rule)
    pred.name = "Prophet"
    return pred

def _make_time_features(ts: pd.Series, bucket: str, seasonal_periods: int) -> pd.DataFrame:
    """
    Feature set designed to work for Day/Week/Month buckets.
    Builds lag and rolling features plus calendar features.
    """
    df = pd.DataFrame({"y": ts.astype(float)})
    idx = df.index

    # Calendar features depend on bucket granularity
    df["month"] = idx.month
    df["quarter"] = idx.quarter
    df["year"] = idx.year

    if bucket == "Day":
        df["dow"] = idx.dayofweek
        df["dom"] = idx.day
        df["woy"] = idx.isocalendar().week.astype(int)
    elif bucket == "Week":
        df["woy"] = idx.isocalendar().week.astype(int)
    else:
        df["dom"] = idx.day  # mostly 1 for MS, but harmless

    # Lags
    for L in [1, 2, 3, 4]:
        df[f"lag_{L}"] = df["y"].shift(L)

    # Seasonal lag
    if seasonal_periods and seasonal_periods >= 2:
        df[f"lag_season_{seasonal_periods}"] = df["y"].shift(seasonal_periods)

    # Rolling stats
    for w in [3, 6, 12]:
        df[f"roll_mean_{w}"] = df["y"].rolling(w).mean()
        df[f"roll_std_{w}"] = df["y"].rolling(w).std()

    return df

def xgboost_forecast(y: pd.Series, horizon: int, rule: str, bucket: str,
                     seasonal_periods: int, n_estimators: int, max_depth: int,
                     learning_rate: float) -> pd.Series:
    if not _HAS_XGB:
        raise RuntimeError("xgboost is not available in this environment.")

    y_hist = y.astype(float).copy()

    feat = _make_time_features(y_hist, bucket=bucket, seasonal_periods=seasonal_periods).dropna()
    if len(feat) < 25:
        raise RuntimeError("Not enough history after lag/rolling features for XGBoost (need ~25+ rows).")

    X = feat.drop(columns=["y"])
    y_train = feat["y"]

    model = xgb.XGBRegressor(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        learning_rate=float(learning_rate),
        objective="reg:squarederror",
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    model.fit(X, y_train)

    idx_future = _future_index(y_hist.index[-1], horizon, rule)
    preds = []

    history = y_hist.copy()
    for ts in idx_future:
        tmp_feat = _make_time_features(history, bucket=bucket, seasonal_periods=seasonal_periods).iloc[[-1]]
        x_last = tmp_feat.drop(columns=["y"])
        yhat = float(model.predict(x_last)[0])
        preds.append(yhat)
        history.loc[ts] = yhat

    return pd.Series(preds, index=idx_future, name="XGBoost")


# ---------------- Data utilities ----------------
def load_sample():
    # Expect you to ship a file in your repo at data/sample_weekly_retail.csv
    return pd.read_csv("data/sample_weekly_retail.csv")

def _coerce_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=False)

def validate_and_prepare(df: pd.DataFrame, date_col: str, y_col: str, bucket: str) -> pd.Series:
    out = df.copy()

    if date_col not in out.columns or y_col not in out.columns:
        raise ValueError(f"Missing columns. Found: {list(out.columns)}")

    out[date_col] = _coerce_datetime(out[date_col])
    out[y_col] = pd.to_numeric(out[y_col], errors="coerce")

    out = out.dropna(subset=[date_col, y_col]).sort_values(date_col)
    out = out.set_index(date_col)

    # Resample into selected bucket using sum (typical demand aggregation)
    rule = BUCKET_TO_RULE[bucket]
    y = out[y_col].resample(rule).sum()

    # Fill gaps sensibly
    # For demand, missing buckets often mean zero; but if your data is sparse sampling, interpolation helps.
    # Here: use 0 fill then smooth interpolate for stability.
    y = y.asfreq(rule)
    y = y.fillna(0.0)
    # Avoid introducing negatives, keep it simple:
    y = y.astype(float)

    # Optional light smoothing for extremely noisy data (comment out if undesired)
    # y = y.rolling(2, min_periods=1).mean()

    y.name = "demand"
    return y


@st.cache_data(show_spinner=False)
def run_models(y: pd.Series, horizon: int, cfg: dict):
    """
    Uses last `horizon` as test window; train is the rest.
    Returns train/test, forecast dict, metrics df, errors dict.
    """
    if horizon < 1:
        raise ValueError("Horizon must be >= 1")
    if len(y) < max(8, horizon + 4):
        raise ValueError("Time series too short for selected horizon.")

    y = y.sort_index()
    y_train = y.iloc[:-horizon].copy()
    y_test = y.iloc[-horizon:].copy()

    forecasts = {}
    timings = {}
    errors = {}

    def _timeit(name, fn):
        t0 = time.time()
        try:
            fc = fn()
            # Align forecast index to test index if possible (same bucket)
            forecasts[name] = fc
        except Exception as e:
            errors[name] = str(e)
        timings[name] = time.time() - t0

    rule = cfg["rule"]
    bucket = cfg["bucket"]
    sp = cfg["seasonal_periods"]

    # Baselines
    if cfg["use_naive"]:
        _timeit("Naive", lambda: naive_forecast(y_train, horizon, rule))

    if cfg["use_snaive"]:
        _timeit("Seasonal Naive", lambda: seasonal_naive_forecast(y_train, horizon, rule, sp))

    if cfg["use_ma"]:
        _timeit("Moving Average", lambda: moving_average_level_forecast(y_train, horizon, rule, cfg["ma_window"]))

    # ETS/HW family
    if cfg["use_ets"]:
        _timeit("ETS", lambda: ets_forecast(
            y_train,
            horizon=horizon,
            rule=rule,
            seasonal_periods=sp,
            seasonal_mode=cfg["ets_seasonal"],
            trend_mode=cfg["ets_trend"],
        ))

    if cfg["use_prophet"]:
        _timeit("Prophet", lambda: prophet_forecast(
            y_train, horizon=horizon, rule=rule, seasonality_mode=cfg["prophet_mode"]
        ))

    if cfg["use_xgb"]:
        _timeit("XGBoost", lambda: xgboost_forecast(
            y_train,
            horizon=horizon,
            rule=rule,
            bucket=bucket,
            seasonal_periods=sp,
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
                "Train Time (s)": float(timings.get(name, np.nan)),
            })
        else:
            rows.append({
                "Algorithm": name,
                "MAPE (%)": np.nan,
                "RMSE": np.nan,
                "MAE": np.nan,
                "Bias": np.nan,
                "Train Time (s)": float(timings.get(name, np.nan)),
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
    <span class='metric-pill'>Naive</span>
    <span class='metric-pill'>Seasonal Naive</span>
    <span class='metric-pill'>Moving Average</span>
    <span class='metric-pill'>ETS (Holt-Winters)</span>
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

# Bucket selection (default suggested from data after load)
bucket_default = "Day"
if df_raw is not None and date_col in df_raw.columns:
    try:
        dt_tmp = pd.to_datetime(df_raw[date_col], errors="coerce").dropna()
        if len(dt_tmp) >= 3:
            bucket_default = _suggest_bucket_from_dates(pd.DatetimeIndex(dt_tmp))
    except Exception:
        bucket_default = "Day"

st.sidebar.markdown("### 🪣 Forecast bucket")
bucket = st.sidebar.selectbox("Aggregate & forecast by", ["Day", "Week", "Month"], index=["Day","Week","Month"].index(bucket_default))
rule = BUCKET_TO_RULE[bucket]
seasonal_periods = _seasonal_period_for_bucket(bucket)

st.sidebar.markdown("### 🔮 Forecast settings")
horizon = st.sidebar.number_input(f"Forecast horizon ({bucket.lower()} buckets)", min_value=1, max_value=365, value=12, step=1)

st.sidebar.markdown("### 🧠 Algorithms")

use_naive = st.sidebar.checkbox("Naive", value=True)
use_snaive = st.sidebar.checkbox("Seasonal Naive", value=True)

use_ma = st.sidebar.checkbox("Moving Average (level)", value=True)
ma_window = st.sidebar.selectbox("MA window (buckets)", [2, 3, 4, 6, 12, 26, 52], index=3, disabled=not use_ma)

use_ets = st.sidebar.checkbox("ETS (Exponential Smoothing)", value=True, disabled=not _HAS_STATSMODELS)
ets_trend = st.sidebar.selectbox("ETS trend", ["add", "mul", "None"], index=0, disabled=not (use_ets and _HAS_STATSMODELS))
ets_seasonal = st.sidebar.selectbox("ETS seasonal", ["add", "mul"], index=0, disabled=not (use_ets and _HAS_STATSMODELS))

use_prophet = st.sidebar.checkbox("Prophet", value=False, disabled=not _HAS_PROPHET)
prophet_mode = st.sidebar.selectbox("Prophet seasonality mode", ["multiplicative", "additive"], index=0, disabled=not (use_prophet and _HAS_PROPHET))

use_xgb = st.sidebar.checkbox("XGBoost", value=True, disabled=not _HAS_XGB)
xgb_estimators = st.sidebar.slider("XGB estimators", 50, 800, 300, 50, disabled=not (use_xgb and _HAS_XGB))
xgb_depth = st.sidebar.slider("XGB max_depth", 2, 12, 5, 1, disabled=not (use_xgb and _HAS_XGB))
xgb_lr = st.sidebar.slider("XGB learning_rate", 0.01, 0.3, 0.05, 0.01, disabled=not (use_xgb and _HAS_XGB))

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div style='color: #A0AEC0; font-size: 12px; padding: 0.5rem 0 0.5rem 0;'>
    <strong style='color: #E2E8F0;'>Bucket:</strong> {bucket} ({rule})<br>
    <strong style='color: #E2E8F0;'>Seasonal period:</strong> {seasonal_periods}
</div>
""", unsafe_allow_html=True)

run_btn = st.sidebar.button("Run Forecasts", type="primary")


# ---------------- Main ----------------
if df_raw is None:
    st.info("Upload a CSV to begin.")
    st.stop()

try:
    y = validate_and_prepare(df_raw, date_col=date_col, y_col=y_col, bucket=bucket)
except Exception as e:
    st.error(f"Data validation failed: {e}")
    st.stop()

st.markdown("## 🧾 Data preview (after bucketing)")
st.dataframe(pd.DataFrame({"demand": y}).tail(24), use_container_width=True)

# Capability pills
caps = [
    f"Bucket: {bucket}",
    f"Rule: {rule}",
    f"Rows: {len(y):,}",
]
if not _HAS_STATSMODELS:
    caps.append("statsmodels: not installed")
if not _HAS_PROPHET:
    caps.append("Prophet: not installed")
if not _HAS_XGB:
    caps.append("XGBoost: not installed")

st.markdown(
    "<div style='margin-top:0.2rem; margin-bottom:1.4rem;'>"
    + "".join([f"<span class='info-pill'>{c}</span>" for c in caps])
    + "</div>",
    unsafe_allow_html=True
)

if run_btn:
    cfg = dict(
        bucket=bucket,
        rule=rule,
        seasonal_periods=int(seasonal_periods),

        use_naive=bool(use_naive),
        use_snaive=bool(use_snaive),
        use_ma=bool(use_ma),
        ma_window=int(ma_window),

        use_ets=bool(use_ets) and _HAS_STATSMODELS,
        ets_trend=(None if ets_trend == "None" else ets_trend),
        ets_seasonal=str(ets_seasonal),

        use_prophet=bool(use_prophet) and _HAS_PROPHET,
        prophet_mode=str(prophet_mode),

        use_xgb=bool(use_xgb) and _HAS_XGB,
        xgb_estimators=int(xgb_estimators),
        xgb_depth=int(xgb_depth),
        xgb_lr=float(xgb_lr),
    )

    with st.spinner("Training models and generating forecasts..."):
        y_train, y_test, forecasts, metrics_df, errors = run_models(y, int(horizon), cfg)

    st.markdown("## 📌 Key metrics")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Train rows", f"{len(y_train):,}")
    with c2:
        st.metric("Test rows", f"{len(y_test):,}")
    with c3:
        st.metric("Horizon", f"{int(horizon)} {bucket.lower()}")
    with c4:
        st.metric("Models run", f"{len(forecasts):,}")

    if len(errors) > 0:
        with st.expander("⚠️ Some models failed to run (click to view)"):
            for k, v in errors.items():
                st.write(f"**{k}**: {v}")

    st.markdown("## 📊 Forecast comparison")
    st.plotly_chart(make_chart(y_train, y_test, forecasts), use_container_width=True, config={"displayModeBar": False})

    st.markdown("## ✅ Accuracy metrics (on holdout window)")
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    st.markdown("## 💾 Export")
    export_df = pd.DataFrame({"actual": y_test})
    for name, fc in forecasts.items():
        export_df[name] = fc.reindex(export_df.index)

    csv = export_df.to_csv(index=True).encode("utf-8")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button("📥 Download Forecasts CSV", data=csv, file_name=f"forecasts_{bucket.lower()}_{ts}.csv", mime="text/csv")

    with st.expander("🔧 Technical details"):
        st.markdown(f"""
        **What this app does**
        - Parses and validates a demand time series (date + demand column)
        - Resamples to a selected bucket (**{bucket}**) using SUM aggregation
        - Splits into train/test using the last **{horizon}** buckets as holdout
        - Runs selected models and compares forecasts
        - Calculates MAPE / RMSE / MAE / Bias on the holdout window
        - Exports a CSV for the forecast window

        **Seasonality defaults**
        - Day → 7
        - Week → 52
        - Month → 12

        **Dependencies**
        - Always: streamlit, pandas, numpy, plotly
        - Optional: statsmodels (ETS), xgboost (XGBoost), prophet (Prophet)
        """)
else:
    st.info("Adjust settings in the sidebar and click **Run Forecasts**.")
