"""
⚡ Household Electric Power Consumption — Prediction Dashboard
================================================================
A production-quality Streamlit analytics dashboard for exploring,
predicting, and forecasting household power consumption.

Run:  streamlit run app.py
"""

# ──────────────────────────────────────────────────────────────
# 1. IMPORTS
# ──────────────────────────────────────────────────────────────
import os
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import zipfile

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# 2. CONSTANTS & PATHS
# ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

CLEANED_DATA_PATH = BASE_DIR / "cleaned_power.parquet"
FEATURED_DATA_PATH = BASE_DIR / "featured_power.parquet"

MODEL_PATHS = {
    "Linear Regression": BASE_DIR / "best_model_lr.pkl",
    "Decision Tree": BASE_DIR / "best_model_dt.pkl",
    "Random Forest": BASE_DIR / "best_model_rf.pkl",
    "XGBoost": BASE_DIR / "best_model_xgb.pkl",
    "LightGBM": BASE_DIR / "best_model_lgb.pkl",
    "CatBoost": BASE_DIR / "best_model_cat.pkl",
}

FORECAST_MODEL_PATHS = {
    "Prophet": BASE_DIR / "prophet.pkl",
    "Auto ARIMA": BASE_DIR / "auto_arima.pkl",
    "Manual ARIMA": BASE_DIR / "manual_arima.pkl",
    "Manual SARIMA": BASE_DIR / "manual_sarima.pkl",
}

# Best regression model for prediction page
BEST_REG_MODEL_KEY = "CatBoost"

# Best forecasting model
BEST_TS_MODEL_KEY = "Prophet"

# Target variable
TARGET = "Global_active_power"

# Features to drop for regression (as in notebook 04)
DROP_COLS = ["Global_active_power", "Global_intensity"]

# Feature display names for prediction form
FEATURE_INFO = {
    "Global_reactive_power": {"label": "Global Reactive Power (kW)", "min": 0.0, "max": 1.4, "default": 0.1, "step": 0.01, "help": "Household global reactive power (kilowatt)"},
    "Voltage": {"label": "Voltage (V)", "min": 220.0, "max": 255.0, "default": 240.0, "step": 0.1, "help": "Average voltage (volt)"},
    "Sub_metering_1": {"label": "Sub Metering 1 (Wh)", "min": 0.0, "max": 88.0, "default": 0.0, "step": 1.0, "help": "Kitchen (dishwasher, oven, microwave)"},
    "Sub_metering_2": {"label": "Sub Metering 2 (Wh)", "min": 0.0, "max": 80.0, "default": 0.0, "step": 1.0, "help": "Laundry (washing machine, dryer, light, fridge)"},
    "Sub_metering_3": {"label": "Sub Metering 3 (Wh)", "min": 0.0, "max": 31.0, "default": 17.0, "step": 1.0, "help": "Electric water heater & air conditioner"},
    "hour": {"label": "Hour (0–23)", "min": 0.0, "max": 23.0, "default": 12.0, "step": 1.0, "help": "Hour of day"},
    "day": {"label": "Day of Month (1–31)", "min": 1.0, "max": 31.0, "default": 15.0, "step": 1.0, "help": "Day of month"},
    "day_of_week": {"label": "Day of Week (0=Mon – 6=Sun)", "min": 0.0, "max": 6.0, "default": 2.0, "step": 1.0, "help": "0=Monday … 6=Sunday"},
    "month": {"label": "Month (1–12)", "min": 1.0, "max": 12.0, "default": 6.0, "step": 1.0, "help": "Month of year"},
    "quarter": {"label": "Quarter (1–4)", "min": 1.0, "max": 4.0, "default": 2.0, "step": 1.0, "help": "Quarter of year"},
    "dayofyear": {"label": "Day of Year (1–366)", "min": 1.0, "max": 366.0, "default": 180.0, "step": 1.0, "help": "Day of year"},
    "weekofyear": {"label": "Week of Year (1–53)", "min": 1.0, "max": 53.0, "default": 26.0, "step": 1.0, "help": "ISO week number"},
    "is_weekend": {"label": "Is Weekend (0/1)", "min": 0.0, "max": 1.0, "default": 0.0, "step": 1.0, "help": "1 if Saturday/Sunday"},
    "lag_1": {"label": "Lag 1 min (kW)", "min": 0.0, "max": 11.0, "default": 1.0, "step": 0.01, "help": "Power 1 minute ago"},
    "lag_60": {"label": "Lag 60 min (kW)", "min": 0.0, "max": 11.0, "default": 1.0, "step": 0.01, "help": "Power 1 hour ago"},
    "lag_1440": {"label": "Lag 1440 min (kW)", "min": 0.0, "max": 11.0, "default": 1.0, "step": 0.01, "help": "Power 1 day ago"},
    "previous_hour_avg": {"label": "Previous Hour Avg (kW)", "min": 0.0, "max": 9.0, "default": 1.0, "step": 0.01, "help": "Rolling 60-min mean"},
    "previous_day_avg": {"label": "Previous Day Avg (kW)", "min": 0.0, "max": 6.0, "default": 1.0, "step": 0.01, "help": "Rolling 1440-min mean"},
    "previous_hour_std": {"label": "Previous Hour Std (kW)", "min": 0.0, "max": 5.0, "default": 0.5, "step": 0.01, "help": "Rolling 60-min std"},
    "previous_day_std": {"label": "Previous Day Std (kW)", "min": 0.0, "max": 4.0, "default": 0.5, "step": 0.01, "help": "Rolling 1440-min std"},
}

# Pre-computed metrics from final evaluation notebook
REGRESSION_METRICS = pd.DataFrame({
    "Model": ["Linear Regression", "Decision Tree", "Random Forest",
              "XGBoost", "LightGBM", "CatBoost"],
    "MAE":  [0.7102, 0.0663, 0.0602, 0.0629, 0.0631, 0.0639],
    "RMSE": [0.8826, 0.1738, 0.1551, 0.1562, 0.1497, 0.1480],
    "R²":   [-0.0135, 0.9607, 0.9687, 0.9682, 0.9708, 0.9715],
})

FORECAST_METRICS = pd.DataFrame({
    "Model": ["Manual ARIMA", "Auto ARIMA", "Manual SARIMA", "Prophet"],
    "AIC":  [51775.82, 53974.67, 48235.51, np.nan],
    "MAE":  [0.5775, 0.6241, 0.5065, 0.4884],
    "RMSE": [0.7011, 0.7364, 0.6832, 0.6388],
    "MAPE (%)": [99.62, 108.58, 70.93, 73.37],
})


# ──────────────────────────────────────────────────────────────
# 3. CUSTOM THEME CSS
# ──────────────────────────────────────────────────────────────
def inject_custom_css():
    """Inject global CSS for a polished dark dashboard aesthetic."""
    st.markdown("""
    <style>
    /* ── Import Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ── Root variables ── */
    :root {
        --bg-primary:   #0e1117;
        --bg-card:      #161b22;
        --bg-card-alt:  #1c2333;
        --border:       #30363d;
        --accent:       #58a6ff;
        --accent-glow:  rgba(88,166,255,0.15);
        --success:      #3fb950;
        --warning:      #d29922;
        --danger:       #f85149;
        --text-primary: #e6edf3;
        --text-muted:   #8b949e;
        --radius:       12px;
    }

    /* ── Global typography ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }

    /* ── Sidebar styling ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] .stRadio > label {
        font-weight: 600;
        letter-spacing: 0.02em;
    }

    /* ── Metric card ── */
    .metric-card {
        background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-card-alt) 100%);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.25rem 1.5rem;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.35);
    }
    .metric-card .metric-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: var(--accent);
        margin-bottom: 0.25rem;
    }
    .metric-card .metric-label {
        font-size: 0.82rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    /* ── Hero / Banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #0d1b2a 0%, #1b2838 40%, #162447 100%);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 2.5rem 2rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -50%; right: -20%;
        width: 500px; height: 500px;
        background: radial-gradient(circle, rgba(88,166,255,0.08) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-banner h1 {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #58a6ff, #79c0ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-banner p {
        color: var(--text-muted);
        font-size: 1.05rem;
        line-height: 1.6;
    }

    /* ── Section header ── */
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--text-primary);
        border-left: 3px solid var(--accent);
        padding-left: 0.75rem;
        margin: 1.5rem 0 1rem;
    }

    /* ── Info strip ── */
    .info-strip {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
        color: var(--text-muted);
        font-size: 0.9rem;
        line-height: 1.65;
    }

    /* ── Workflow card ── */
    .workflow-step {
        display: inline-block;
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.6rem 1rem;
        margin: 0.25rem;
        font-size: 0.82rem;
        font-weight: 500;
        color: var(--text-primary);
        transition: background 0.2s;
    }
    .workflow-step:hover { background: var(--bg-card-alt); }
    .workflow-arrow {
        display: inline-block;
        color: var(--accent);
        font-size: 1.1rem;
        margin: 0 0.15rem;
        vertical-align: middle;
    }

    /* ── Tech badge ── */
    .tech-badge {
        display: inline-block;
        background: var(--accent-glow);
        border: 1px solid rgba(88,166,255,0.3);
        color: var(--accent);
        border-radius: 20px;
        padding: 0.3rem 0.85rem;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 0.2rem;
    }

    /* ── Prediction result ── */
    .prediction-box {
        background: linear-gradient(135deg, #0d2137 0%, #132a46 100%);
        border: 1px solid rgba(88,166,255,0.3);
        border-radius: var(--radius);
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .prediction-box .pred-value {
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(135deg, #58a6ff, #3fb950);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .prediction-box .pred-label {
        color: var(--text-muted);
        font-size: 0.9rem;
        margin-top: 0.3rem;
    }

    /* ── Best model highlight ── */
    .best-badge {
        display: inline-block;
        background: rgba(63,185,80,0.15);
        border: 1px solid rgba(63,185,80,0.4);
        color: var(--success);
        border-radius: 6px;
        padding: 0.15rem 0.55rem;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* ── Insight card ── */
    .insight-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.25rem 1.5rem;
        margin-bottom: 0.75rem;
    }
    .insight-card h4 {
        color: var(--accent);
        margin-bottom: 0.5rem;
        font-size: 1rem;
    }
    .insight-card p, .insight-card li {
        color: var(--text-muted);
        font-size: 0.88rem;
        line-height: 1.65;
    }

    /* ── Hide Streamlit branding ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ── Dataframe styling ── */
    .stDataFrame { border-radius: var(--radius) !important; }

    /* ── Tab underline accent ── */
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: var(--accent) !important;
    }
    </style>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# 4. DATA LOADING (CACHED)
# ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_cleaned_data() -> pd.DataFrame:
    """Load the cleaned parquet dataset."""
    if not CLEANED_DATA_PATH.exists():
        st.error(f"Cleaned data not found at `{CLEANED_DATA_PATH}`")
        st.stop()
    return pd.read_parquet(CLEANED_DATA_PATH)


@st.cache_data(show_spinner=False)
def load_featured_data() -> pd.DataFrame:
    """Load the feature-engineered parquet dataset."""
    if not FEATURED_DATA_PATH.exists():
        st.error(f"Featured data not found at `{FEATURED_DATA_PATH}`")
        st.stop()
    return pd.read_parquet(FEATURED_DATA_PATH)


# ──────────────────────────────────────────────────────────────
# 5. MODEL LOADING (CACHED)
# ──────────────────────────────────────────────────────────────
def extract_if_needed(path):
    """
    Extract corresponding ZIP if the PKL file is missing.
    """

    if path.exists():
        return

    zip_path = path.with_suffix(".zip")

    if zip_path.exists():
        print(f"Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(path.parent)

@st.cache_resource(show_spinner=False)
def load_regression_model(key: str):
    """Load a saved regression model by key name."""
    path = MODEL_PATHS.get(key)
    if path is None or not path.exists():
        return None
    return joblib.load(path)


@st.cache_resource(show_spinner=False)
def load_forecast_model(key: str):
    """Load a saved forecasting model by key name."""
    path = FORECAST_MODEL_PATHS.get(key)
    if path is None:
        return None

    if key in ["Manual ARIMA", "Auto ARIMA"]:
        extract_if_needed(path)

    if not path.exists():
        return None

    return joblib.load(path)


# ──────────────────────────────────────────────────────────────
# 6. HELPER / UTILITY FUNCTIONS
# ──────────────────────────────────────────────────────────────
def metric_card(label: str, value: str, col=None):
    """Render a styled metric card inside an optional column."""
    html = f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """
    if col is not None:
        col.markdown(html, unsafe_allow_html=True)
    else:
        st.markdown(html, unsafe_allow_html=True)


def section_header(text: str):
    """Render a styled section header."""
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)


def set_plot_style():
    """Apply dark plot style for matplotlib / seaborn figures."""
    plt.style.use("dark_background")
    plt.rcParams.update({
        "figure.facecolor": "#0e1117",
        "axes.facecolor": "#161b22",
        "axes.edgecolor": "#30363d",
        "axes.labelcolor": "#8b949e",
        "xtick.color": "#8b949e",
        "ytick.color": "#8b949e",
        "text.color": "#e6edf3",
        "grid.color": "#21262d",
        "grid.alpha": 0.5,
        "font.family": "sans-serif",
        "figure.figsize": (10, 5),
    })


def compute_cyclical(value, period):
    """Return (sin, cos) encoding for a cyclical feature."""
    sin_val = np.sin(2 * np.pi * value / period)
    cos_val = np.cos(2 * np.pi * value / period)
    return float(sin_val), float(cos_val)


# ──────────────────────────────────────────────────────────────
# 7. PAGE IMPLEMENTATIONS
# ──────────────────────────────────────────────────────────────

# ────────── 7-A  HOME ──────────
def page_home():
    # Hero
    st.markdown("""
    <div class="hero-banner">
        <h1>⚡ Household Electric Power Consumption</h1>
        <p>
            An end-to-end Machine Learning &amp; Time Series Forecasting project that
            predicts household energy usage using advanced regression and forecasting models.
            Explore interactive analytics, generate real-time predictions, and compare
            model performance — all from this single dashboard.
        </p>
        <p style="margin-top:0.75rem; font-size:0.85rem; color:#58a6ff;">
            By <strong>Saikat Sarkar</strong> · B.S. Data Science &amp; AI · IIT Jodhpur
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Quick stats
    section_header("Project at a Glance")
    c1, c2, c3, c4 = st.columns(4)
    metric_card("Total Records", "2,075,259", c1)
    metric_card("Features Engineered", "28", c2)
    metric_card("Regression Models", "6", c3)
    metric_card("Forecasting Models", "4", c4)

    st.markdown("<br>", unsafe_allow_html=True)

    c5, c6, c7, c8 = st.columns(4)
    metric_card("Best Regression R²", "0.9715", c5)
    metric_card("Best Reg. Model", "CatBoost", c6)
    metric_card("Best Forecast RMSE", "0.6388", c7)
    metric_card("Best TS Model", "Prophet", c8)

    st.markdown("<br>", unsafe_allow_html=True)

    # Project objectives
    section_header("Project Objectives")
    st.markdown("""
    <div class="info-strip">
        <strong>1.</strong> Clean &amp; pre-process the UCI Household Power Consumption dataset.<br>
        <strong>2.</strong> Perform thorough Exploratory Data Analysis to uncover temporal patterns.<br>
        <strong>3.</strong> Engineer meaningful temporal, lag, and rolling-window features.<br>
        <strong>4.</strong> Train &amp; tune multiple regression models to predict Global Active Power.<br>
        <strong>5.</strong> Apply ARIMA, SARIMA &amp; Prophet for time series forecasting.<br>
        <strong>6.</strong> Compare all models and present actionable insights.
    </div>
    """, unsafe_allow_html=True)

    # Workflow
    section_header("Project Workflow")
    steps = [
        "Data Cleaning", "EDA", "Feature Engineering",
        "Regression Models", "Time Series Forecasting", "Final Evaluation",
    ]
    workflow_html = ""
    for i, step in enumerate(steps):
        workflow_html += f'<span class="workflow-step">{step}</span>'
        if i < len(steps) - 1:
            workflow_html += '<span class="workflow-arrow">→</span>'
    st.markdown(workflow_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Technologies
    section_header("Technologies & Libraries")
    techs = [
        "Python", "Pandas", "NumPy", "Matplotlib", "Seaborn",
        "Scikit-Learn", "XGBoost", "LightGBM", "CatBoost",
        "Statsmodels", "pmdarima", "Prophet", "Streamlit", "Joblib",
    ]
    tech_html = " ".join(f'<span class="tech-badge">{t}</span>' for t in techs)
    st.markdown(tech_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Dataset summary
    section_header("Dataset Summary")
    st.markdown("""
    <div class="info-strip">
        <strong>Source:</strong> UCI Machine Learning Repository — Individual Household Electric Power Consumption.<br>
        <strong>Period:</strong> December 2006 — November 2010 (≈ 4 years of minute-level readings).<br>
        <strong>Granularity:</strong> 1-minute sampling rate.<br>
        <strong>Variables:</strong> Global Active Power, Reactive Power, Voltage, Global Intensity, Sub-metering 1/2/3.
    </div>
    """, unsafe_allow_html=True)


# ────────── 7-B  DATASET OVERVIEW ──────────
def page_dataset():
    st.markdown("## Dataset Overview")

    with st.spinner("Loading dataset …"):
        df_clean = load_cleaned_data()
        df_feat = load_featured_data()

    tab1, tab2 = st.tabs(["Cleaned Dataset", "Feature-Engineered Dataset"])

    for label, df in zip(["Cleaned Dataset", "Feature-Engineered Dataset"],
                         [df_clean, df_feat]):
        with tab1 if label.startswith("Cleaned") else tab2:
            section_header(f"{label} — Shape & Info")

            c1, c2, c3 = st.columns(3)
            metric_card("Rows", f"{df.shape[0]:,}", c1)
            metric_card("Columns", str(df.shape[1]), c2)
            mem = df.memory_usage(deep=True).sum() / 1e6
            metric_card("Memory", f"{mem:.1f} MB", c3)

            st.markdown("<br>", unsafe_allow_html=True)

            with st.expander("Column Information", expanded=False):
                col_info = pd.DataFrame({
                    "Column": df.columns,
                    "Dtype": [str(dt) for dt in df.dtypes],
                    "Non-Null": [int(df[c].notna().sum()) for c in df.columns],
                    "Null": [int(df[c].isna().sum()) for c in df.columns],
                })
                st.dataframe(col_info, use_container_width=True, hide_index=True)

            with st.expander("Descriptive Statistics", expanded=False):
                st.dataframe(df.describe().round(4), use_container_width=True)

            with st.expander("Sample Data (first 100 rows)", expanded=False):
                st.dataframe(df.head(100), use_container_width=True)

            with st.expander("Missing Values", expanded=False):
                missing = df.isnull().sum()
                missing = missing[missing > 0]
                if missing.empty:
                    st.success("No missing values in this dataset.")
                else:
                    st.dataframe(
                        pd.DataFrame({"Column": missing.index, "Missing": missing.values}),
                        use_container_width=True, hide_index=True,
                    )


# ────────── 7-C  EXPLORATORY DATA ANALYSIS ──────────
def page_eda():
    st.markdown("## Exploratory Data Analysis")

    with st.spinner("Loading data for analysis …"):
        df = load_cleaned_data()

    set_plot_style()

    # Plot selector
    plot_options = [
        "Target Distribution",
        "Correlation Heatmap",
        "Monthly Trend",
        "Hourly Trend",
        "Daily Trend (Day of Week)",
        "Weekly Trend (Week of Year)",
        "Seasonal Analysis",
        "Rolling Mean (24h)",
        "Rolling Std (24h)",
    ]
    selected = st.selectbox("Select Visualization", plot_options, index=0)

    fig, ax = plt.subplots(figsize=(10, 5))

    if selected == "Target Distribution":
        ax.hist(df[TARGET].dropna(), bins=80, color="#58a6ff", edgecolor="#0e1117", alpha=0.85)
        ax.set_xlabel("Global Active Power (kW)")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Global Active Power", fontweight="bold")

    elif selected == "Correlation Heatmap":
        fig, ax = plt.subplots(figsize=(10, 8))
        corr = df.corr(numeric_only=True)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                    linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
        ax.set_title("Correlation Heatmap", fontweight="bold")

    elif selected == "Monthly Trend":
        monthly = df.groupby(df.index.month)[TARGET].mean()
        ax.plot(monthly.index, monthly.values, marker="o", color="#58a6ff", linewidth=2)
        ax.fill_between(monthly.index, monthly.values, alpha=0.15, color="#58a6ff")
        ax.set_xlabel("Month")
        ax.set_ylabel("Avg Global Active Power (kW)")
        ax.set_title("Average Power Consumption by Month", fontweight="bold")
        ax.set_xticks(range(1, 13))

    elif selected == "Hourly Trend":
        hourly = df.groupby(df.index.hour)[TARGET].mean()
        ax.plot(hourly.index, hourly.values, marker="o", color="#3fb950", linewidth=2)
        ax.fill_between(hourly.index, hourly.values, alpha=0.15, color="#3fb950")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Avg Global Active Power (kW)")
        ax.set_title("Average Power Consumption by Hour", fontweight="bold")
        ax.set_xticks(range(0, 24))

    elif selected == "Daily Trend (Day of Week)":
        daily = df.groupby(df.index.dayofweek)[TARGET].mean()
        day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        ax.bar(range(7), daily.values, color="#79c0ff", edgecolor="#0e1117")
        ax.set_xticks(range(7))
        ax.set_xticklabels(day_labels)
        ax.set_ylabel("Avg Global Active Power (kW)")
        ax.set_title("Average Power Consumption by Day of Week", fontweight="bold")

    elif selected == "Weekly Trend (Week of Year)":
        weekly = df.groupby(df.index.isocalendar().week.astype(int))[TARGET].mean()
        ax.plot(weekly.index, weekly.values, color="#d29922", linewidth=1.5)
        ax.fill_between(weekly.index, weekly.values, alpha=0.12, color="#d29922")
        ax.set_xlabel("Week of Year")
        ax.set_ylabel("Avg Global Active Power (kW)")
        ax.set_title("Average Power Consumption by Week", fontweight="bold")

    elif selected == "Seasonal Analysis":
        df_temp = df.copy()
        df_temp["month"] = df_temp.index.month
        df_temp["season"] = df_temp["month"].map(
            {12: "Winter", 1: "Winter", 2: "Winter",
             3: "Spring", 4: "Spring", 5: "Spring",
             6: "Summer", 7: "Summer", 8: "Summer",
             9: "Autumn", 10: "Autumn", 11: "Autumn"}
        )
        season_order = ["Winter", "Spring", "Summer", "Autumn"]
        season_avg = df_temp.groupby("season")[TARGET].mean().reindex(season_order)
        colors = ["#58a6ff", "#3fb950", "#f85149", "#d29922"]
        ax.bar(season_order, season_avg.values, color=colors, edgecolor="#0e1117")
        ax.set_ylabel("Avg Global Active Power (kW)")
        ax.set_title("Seasonal Power Consumption", fontweight="bold")

    elif selected == "Rolling Mean (24h)":
        # Sample for performance (every 60th minute = hourly)
        sampled = df[TARGET].iloc[::60]
        rolling = sampled.rolling(window=24).mean()
        ax.plot(rolling.index, rolling.values, color="#58a6ff", linewidth=0.8)
        ax.set_xlabel("Date")
        ax.set_ylabel("Rolling 24h Mean (kW)")
        ax.set_title("24-Hour Rolling Mean of Power Consumption", fontweight="bold")
        fig.autofmt_xdate()

    elif selected == "Rolling Std (24h)":
        sampled = df[TARGET].iloc[::60]
        rolling = sampled.rolling(window=24).std()
        ax.plot(rolling.index, rolling.values, color="#f85149", linewidth=0.8)
        ax.set_xlabel("Date")
        ax.set_ylabel("Rolling 24h Std (kW)")
        ax.set_title("24-Hour Rolling Std Dev of Power Consumption", fontweight="bold")
        fig.autofmt_xdate()

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ────────── 7-D  REGRESSION PREDICTION ──────────
def page_regression():
    st.markdown("## Regression — Predict Global Active Power")

    # Load model
    with st.spinner(f"Loading {BEST_REG_MODEL_KEY} model …"):
        model = load_regression_model(BEST_REG_MODEL_KEY)

    if model is None:
        st.error(f"Could not load the {BEST_REG_MODEL_KEY} model. "
                 f"Ensure `{MODEL_PATHS[BEST_REG_MODEL_KEY]}` exists.")
        st.stop()

    st.success(f"**{BEST_REG_MODEL_KEY}** model loaded successfully  ·  R² = 0.9715  ·  RMSE = 0.1480")

    section_header("Enter Feature Values")

    st.markdown("""
    <div class="info-strip">
        Fill in the feature values below to generate a prediction for <strong>Global Active Power (kW)</strong>.
        Cyclical encodings (sin/cos) are computed automatically from the base features.
    </div>
    """, unsafe_allow_html=True)

    # Feature input form
    with st.form("prediction_form"):
        # Row 1: Electrical features
        st.markdown("##### Electrical Measurements")
        r1c1, r1c2, r1c3, r1c4, r1c5 = st.columns(5)
        grp = FEATURE_INFO["Global_reactive_power"]
        global_rp = r1c1.number_input(grp["label"], min_value=grp["min"], max_value=grp["max"],
                                       value=grp["default"], step=grp["step"], help=grp["help"])
        vinfo = FEATURE_INFO["Voltage"]
        voltage = r1c2.number_input(vinfo["label"], min_value=vinfo["min"], max_value=vinfo["max"],
                                     value=vinfo["default"], step=vinfo["step"], help=vinfo["help"])
        s1 = FEATURE_INFO["Sub_metering_1"]
        sub1 = r1c3.number_input(s1["label"], min_value=s1["min"], max_value=s1["max"],
                                  value=s1["default"], step=s1["step"], help=s1["help"])
        s2 = FEATURE_INFO["Sub_metering_2"]
        sub2 = r1c4.number_input(s2["label"], min_value=s2["min"], max_value=s2["max"],
                                  value=s2["default"], step=s2["step"], help=s2["help"])
        s3 = FEATURE_INFO["Sub_metering_3"]
        sub3 = r1c5.number_input(s3["label"], min_value=s3["min"], max_value=s3["max"],
                                  value=s3["default"], step=s3["step"], help=s3["help"])

        # Row 2: Temporal features
        st.markdown("##### Temporal Features")
        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        h = FEATURE_INFO["hour"]
        hour_val = r2c1.number_input(h["label"], min_value=h["min"], max_value=h["max"],
                                      value=h["default"], step=h["step"], help=h["help"])
        d = FEATURE_INFO["day"]
        day_val = r2c2.number_input(d["label"], min_value=d["min"], max_value=d["max"],
                                     value=d["default"], step=d["step"], help=d["help"])
        dw = FEATURE_INFO["day_of_week"]
        dow_val = r2c3.number_input(dw["label"], min_value=dw["min"], max_value=dw["max"],
                                     value=dw["default"], step=dw["step"], help=dw["help"])
        m = FEATURE_INFO["month"]
        month_val = r2c4.number_input(m["label"], min_value=m["min"], max_value=m["max"],
                                       value=m["default"], step=m["step"], help=m["help"])

        r3c1, r3c2, r3c3, r3c4 = st.columns(4)
        q = FEATURE_INFO["quarter"]
        quarter_val = r3c1.number_input(q["label"], min_value=q["min"], max_value=q["max"],
                                         value=q["default"], step=q["step"], help=q["help"])
        dy = FEATURE_INFO["dayofyear"]
        doy_val = r3c2.number_input(dy["label"], min_value=dy["min"], max_value=dy["max"],
                                     value=dy["default"], step=dy["step"], help=dy["help"])
        wy = FEATURE_INFO["weekofyear"]
        woy_val = r3c3.number_input(wy["label"], min_value=wy["min"], max_value=wy["max"],
                                     value=wy["default"], step=wy["step"], help=wy["help"])
        we = FEATURE_INFO["is_weekend"]
        iswe_val = r3c4.number_input(we["label"], min_value=we["min"], max_value=we["max"],
                                      value=we["default"], step=we["step"], help=we["help"])

        # Row 3: Lag & rolling features
        st.markdown("##### Lag & Rolling Features")
        r4c1, r4c2, r4c3 = st.columns(3)
        l1 = FEATURE_INFO["lag_1"]
        lag1 = r4c1.number_input(l1["label"], min_value=l1["min"], max_value=l1["max"],
                                  value=l1["default"], step=l1["step"], help=l1["help"])
        l60 = FEATURE_INFO["lag_60"]
        lag60 = r4c2.number_input(l60["label"], min_value=l60["min"], max_value=l60["max"],
                                   value=l60["default"], step=l60["step"], help=l60["help"])
        l1440 = FEATURE_INFO["lag_1440"]
        lag1440 = r4c3.number_input(l1440["label"], min_value=l1440["min"], max_value=l1440["max"],
                                     value=l1440["default"], step=l1440["step"], help=l1440["help"])

        r5c1, r5c2, r5c3, r5c4 = st.columns(4)
        pha = FEATURE_INFO["previous_hour_avg"]
        prev_h_avg = r5c1.number_input(pha["label"], min_value=pha["min"], max_value=pha["max"],
                                        value=pha["default"], step=pha["step"], help=pha["help"])
        pda = FEATURE_INFO["previous_day_avg"]
        prev_d_avg = r5c2.number_input(pda["label"], min_value=pda["min"], max_value=pda["max"],
                                        value=pda["default"], step=pda["step"], help=pda["help"])
        phs = FEATURE_INFO["previous_hour_std"]
        prev_h_std = r5c3.number_input(phs["label"], min_value=phs["min"], max_value=phs["max"],
                                        value=phs["default"], step=phs["step"], help=phs["help"])
        pds = FEATURE_INFO["previous_day_std"]
        prev_d_std = r5c4.number_input(pds["label"], min_value=pds["min"], max_value=pds["max"],
                                        value=pds["default"], step=pds["step"], help=pds["help"])

        submitted = st.form_submit_button("⚡  Generate Prediction", use_container_width=True)

    if submitted:
        # Compute cyclical
        hour_sin, hour_cos = compute_cyclical(hour_val, 24)
        day_sin, day_cos = compute_cyclical(dow_val, 7)
        month_sin, month_cos = compute_cyclical(month_val, 12)

        # Build feature vector (must match the 26-feature order from notebook)
        feature_order = [
            "Global_reactive_power", "Voltage",
            "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
            "hour", "day", "day_of_week", "month", "quarter",
            "dayofyear", "weekofyear", "is_weekend",
            "hour_sin", "hour_cos", "day_sin", "day_cos",
            "month_sin", "month_cos",
            "lag_1", "lag_60", "lag_1440",
            "previous_hour_avg", "previous_day_avg",
            "previous_hour_std", "previous_day_std",
        ]
        values = [
            global_rp, voltage, sub1, sub2, sub3,
            hour_val, day_val, dow_val, month_val, quarter_val,
            doy_val, woy_val, iswe_val,
            hour_sin, hour_cos, day_sin, day_cos,
            month_sin, month_cos,
            lag1, lag60, lag1440,
            prev_h_avg, prev_d_avg, prev_h_std, prev_d_std,
        ]

        input_df = pd.DataFrame([values], columns=feature_order).astype("float32")

        # Progress animation
        progress = st.progress(0, text="Running prediction …")
        for i in range(100):
            time.sleep(0.005)
            progress.progress(i + 1)
        progress.empty()

        try:
            prediction = model.predict(input_df)[0]
            prediction = max(0, float(prediction))  # Clamp non-negative

            # Confidence message
            if prediction < 1:
                conf_msg = "Low consumption — typical idle or night-time usage."
                conf_icon = "💤"
            elif prediction < 3:
                conf_msg = "Moderate consumption — normal daytime activity."
                conf_icon = "✅"
            else:
                conf_msg = "High consumption — heavy appliance usage detected."
                conf_icon = "🔥"

            st.markdown(f"""
            <div class="prediction-box">
                <div class="pred-value">{prediction:.4f} kW</div>
                <div class="pred-label">Predicted Global Active Power</div>
            </div>
            """, unsafe_allow_html=True)

            st.info(f"{conf_icon}  **{conf_msg}**")

            # Feature summary
            with st.expander("Feature Values Summary", expanded=False):
                st.dataframe(input_df.T.rename(columns={0: "Value"}).round(4),
                             use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# ────────── 7-E  TIME SERIES FORECASTING ──────────
def page_forecasting():
    st.markdown("## Time Series Forecasting")

    section_header("Select Forecast Configuration")

    fc1, fc2 = st.columns([1, 2])
    with fc1:
        model_choice = st.selectbox(
            "Forecasting Model",
            list(FORECAST_MODEL_PATHS.keys()),
            index=list(FORECAST_MODEL_PATHS.keys()).index(BEST_TS_MODEL_KEY),
        )
    with fc2:
        horizon_choice = st.selectbox(
            "Forecast Horizon",
            ["Next 24 Hours", "Next 7 Days", "Custom"],
        )

    if horizon_choice == "Custom":
        periods = st.number_input("Custom periods (hours)", min_value=1,
                                   max_value=720, value=48, step=1)
    elif horizon_choice == "Next 24 Hours":
        periods = 24
    else:
        periods = 7 * 24  # 168 hours

    generate = st.button("Generate Forecast", use_container_width=True, type="primary")

    if generate:
        with st.spinner(f"Loading **{model_choice}** model …"):
            ts_model = load_forecast_model(model_choice)

        if ts_model is None:
            st.error(f"Model file not found for **{model_choice}**. "
                     f"Ensure `{FORECAST_MODEL_PATHS[model_choice]}` exists.")
            st.stop()

        st.success(f"**{model_choice}** loaded successfully.")

        # Progress
        progress = st.progress(0, text="Generating forecast …")

        try:
            if model_choice == "Prophet":
                future = ts_model.make_future_dataframe(
                    periods=periods,
                    freq="h",
                    include_history=False
                )

                for i in range(50):
                    time.sleep(0.01)
                    progress.progress(i + 1)

                forecast = ts_model.predict(future)
                result_df = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
                result_df.columns = ["Datetime", "Forecast", "Lower Bound", "Upper Bound"]

                for i in range(50, 100):
                    time.sleep(0.01)
                    progress.progress(i + 1)

            else:
                # ARIMA / SARIMA models — use .forecast() or .predict()
                for i in range(50):
                    time.sleep(0.01)
                    progress.progress(i + 1)

                forecast_vals = ts_model.predict(n_periods=periods) if hasattr(ts_model, 'predict') and 'n_periods' in str(ts_model.predict.__code__.co_varnames) else ts_model.forecast(steps=periods)

                # Handle different return types
                if isinstance(forecast_vals, tuple):
                    forecast_vals = forecast_vals[0]

                last_date = pd.Timestamp("2010-11-26 21:02:00")
                future_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1),
                                              periods=periods, freq="h")
                result_df = pd.DataFrame({
                    "Datetime": future_dates,
                    "Forecast": forecast_vals,
                })

                for i in range(50, 100):
                    time.sleep(0.01)
                    progress.progress(i + 1)

            progress.empty()

            # Display results
            section_header("Forecast Results")

            # Summary metrics
            mc1, mc2, mc3 = st.columns(3)
            metric_card("Periods", str(periods), mc1)
            metric_card("Mean Forecast", f"{result_df['Forecast'].mean():.3f} kW", mc2)
            metric_card("Max Forecast", f"{result_df['Forecast'].max():.3f} kW", mc3)

            st.markdown("<br>", unsafe_allow_html=True)

            # Chart
            set_plot_style()
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(result_df["Datetime"], result_df["Forecast"],
                    color="#58a6ff", linewidth=1.5, label="Forecast")

            if "Lower Bound" in result_df.columns:
                ax.fill_between(result_df["Datetime"],
                                result_df["Lower Bound"], result_df["Upper Bound"],
                                alpha=0.15, color="#58a6ff", label="Confidence Interval")

            ax.set_xlabel("Datetime")
            ax.set_ylabel("Global Active Power (kW)")
            ax.set_title(f"{model_choice} Forecast — Next {periods} Hours", fontweight="bold")
            ax.legend()
            fig.autofmt_xdate()
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Table
            with st.expander("Forecast Table", expanded=False):
                st.dataframe(result_df.round(4), use_container_width=True, hide_index=True)

            # Download
            csv = result_df.to_csv(index=False)
            st.download_button(
                label="Download Forecast CSV",
                data=csv,
                file_name=f"{model_choice.lower().replace(' ', '_')}_forecast_{periods}h.csv",
                mime="text/csv",
                use_container_width=True,
            )

        except Exception as e:
            progress.empty()
            st.error(f"Forecast generation failed: {e}")
            st.info("This may happen if the model file is incompatible or corrupted. "
                    "Try a different model.")


# ────────── 7-F  MODEL COMPARISON ──────────
def page_comparison():
    st.markdown("## Model Comparison")

    tab_reg, tab_ts = st.tabs(["Regression Models", "Forecasting Models"])

    set_plot_style()

    # ── Regression ──
    with tab_reg:
        section_header("Regression Model Performance")
        df_reg = REGRESSION_METRICS.copy().sort_values("R²", ascending=False).reset_index(drop=True)

        # Highlight best
        best_idx = df_reg["R²"].idxmax()

        def style_reg(row):
            if row.name == best_idx:
                return ["background-color: rgba(63,185,80,0.12)"] * len(row)
            return [""] * len(row)

        styled = df_reg.style.apply(style_reg, axis=1).format({
            "MAE": "{:.4f}", "RMSE": "{:.4f}", "R²": "{:.4f}"
        })
        st.dataframe(styled, use_container_width=True, hide_index=True)

        best_name = df_reg.loc[best_idx, "Model"]
        st.markdown(f'<span class="best-badge">Best Model: {best_name}</span>',
                    unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts
        chart_metric = st.selectbox("Select metric to visualize",
                                     ["R²", "RMSE", "MAE"], key="reg_metric")

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["#3fb950" if m == best_name else "#58a6ff"
                  for m in df_reg["Model"]]
        bars = ax.barh(df_reg["Model"], df_reg[chart_metric], color=colors,
                       edgecolor="#0e1117", height=0.55)
        ax.set_xlabel(chart_metric)
        ax.set_title(f"Regression Models — {chart_metric}", fontweight="bold")
        ax.invert_yaxis()

        # Value labels
        for bar, val in zip(bars, df_reg[chart_metric]):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=9, color="#e6edf3")

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Forecasting ──
    with tab_ts:
        section_header("Forecasting Model Performance")
        df_ts = FORECAST_METRICS.copy().sort_values("RMSE").reset_index(drop=True)

        best_ts_idx = df_ts["RMSE"].idxmin()

        def style_ts(row):
            if row.name == best_ts_idx:
                return ["background-color: rgba(63,185,80,0.12)"] * len(row)
            return [""] * len(row)

        styled_ts = df_ts.style.apply(style_ts, axis=1).format({
            "AIC": "{:.2f}", "MAE": "{:.4f}", "RMSE": "{:.4f}", "MAPE (%)": "{:.2f}%"
        }, na_rep="—")
        st.dataframe(styled_ts, use_container_width=True, hide_index=True)

        best_ts_name = df_ts.loc[best_ts_idx, "Model"]
        st.markdown(f'<span class="best-badge">Best Model: {best_ts_name}</span>',
                    unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        ts_metric = st.selectbox("Select metric to visualize",
                                  ["RMSE", "MAE", "MAPE (%)"], key="ts_metric")

        fig, ax = plt.subplots(figsize=(10, 5))
        plot_df = df_ts.dropna(subset=[ts_metric])
        colors = ["#3fb950" if m == best_ts_name else "#d29922"
                  for m in plot_df["Model"]]
        bars = ax.barh(plot_df["Model"], plot_df[ts_metric], color=colors,
                       edgecolor="#0e1117", height=0.55)
        ax.set_xlabel(ts_metric)
        ax.set_title(f"Forecasting Models — {ts_metric}", fontweight="bold")
        ax.invert_yaxis()

        for bar, val in zip(bars, plot_df[ts_metric]):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=9, color="#e6edf3")

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ────────── 7-G  INSIGHTS ──────────
def page_insights():
    st.markdown("## Key Insights & Findings")

    section_header("Regression Findings")
    st.markdown("""
    <div class="insight-card">
        <h4>Tree-Based Ensembles Dominate</h4>
        <ul>
            <li><strong>CatBoost</strong> emerged as the top performer — lowest RMSE (0.1480) and highest R² (0.9715).</li>
            <li><strong>LightGBM</strong> was a close second (R² = 0.9708, RMSE = 0.1497).</li>
            <li>All tree-based models (DT, RF, XGB, LGBM, CatBoost) significantly outperformed Linear Regression.</li>
            <li><strong>Linear Regression</strong> completely failed (negative R² = -0.0135), confirming that power consumption has strong non-linear relationships.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    section_header("Forecasting Findings")
    st.markdown("""
    <div class="insight-card">
        <h4>Prophet Leads Forecasting</h4>
        <ul>
            <li><strong>Prophet</strong> achieved the lowest RMSE (0.6388) and MAE (0.4884) among all forecasting models.</li>
            <li><strong>Manual SARIMA</strong> was the runner-up and best ARIMA variant (RMSE = 0.6832, AIC = 48,235).</li>
            <li>Seasonal modeling is critical — SARIMA outperformed both Manual and Auto ARIMA thanks to its seasonal component.</li>
            <li>All forecasting models exhibited higher errors than regression models, highlighting the challenge of pure time series prediction.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    section_header("Best Performing Models")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="insight-card" style="text-align:center;">
            <h4>Regression Champion</h4>
            <p style="font-size:1.4rem; font-weight:800; color:#3fb950;">CatBoost</p>
            <p>R² = 0.9715 &nbsp;·&nbsp; RMSE = 0.1480 &nbsp;·&nbsp; MAE = 0.0639</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="insight-card" style="text-align:center;">
            <h4>Forecasting Champion</h4>
            <p style="font-size:1.4rem; font-weight:800; color:#3fb950;">Prophet</p>
            <p>RMSE = 0.6388 &nbsp;·&nbsp; MAE = 0.4884 &nbsp;·&nbsp; MAPE = 73.37%</p>
        </div>
        """, unsafe_allow_html=True)

    section_header("Important Observations")
    st.markdown("""
    <div class="insight-card">
        <h4>Regression vs. Forecasting</h4>
        <ul>
            <li>Regression models achieved dramatically higher accuracy (R² up to 0.97) compared to time series models.</li>
            <li>This highlights that <strong>feature engineering</strong> (lag, rolling stats, cyclical encodings) is a game-changer for power consumption prediction.</li>
            <li>Pure time series models rely solely on historical patterns without external features, limiting their ceiling.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    section_header("Business Insights")
    st.markdown("""
    <div class="insight-card">
        <h4>Actionable Recommendations</h4>
        <ul>
            <li><strong>Peak Hours:</strong> Power consumption peaks during evening hours (17:00–21:00). Energy providers can use this for demand planning.</li>
            <li><strong>Seasonal Patterns:</strong> Winter months show significantly higher consumption — heating drives demand.</li>
            <li><strong>Weekend Effect:</strong> Weekends exhibit different usage patterns, with higher mid-day consumption.</li>
            <li><strong>Deployment:</strong> CatBoost with feature-engineered inputs is recommended for real-time prediction APIs.</li>
            <li><strong>Monitoring:</strong> Rolling statistics (mean & std) serve as excellent anomaly detectors for unusual consumption spikes.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    section_header("Limitations & Future Work")
    st.markdown("""
    <div class="insight-card">
        <h4>Limitations</h4>
        <ul>
            <li>Forecast accuracy degrades for longer horizons; confidence intervals widen significantly.</li>
            <li>Missing external variables (weather, pricing, holidays) limit model performance.</li>
            <li>SARIMA assumes strict periodicity — breaks during anomalous events.</li>
        </ul>
        <h4 style="margin-top:1rem;">Future Improvements</h4>
        <ul>
            <li>Implement deep learning (LSTM, Transformer) for sequence modeling.</li>
            <li>Integrate weather API data for richer feature sets.</li>
            <li>Use Bayesian optimization (Optuna) for hyperparameter tuning.</li>
            <li>Build a FastAPI endpoint for real-time CatBoost predictions.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# ────────── 7-H  ABOUT ──────────
def page_about():
    st.markdown("## About This Project")

    st.markdown("""
    <div class="hero-banner">
        <h1>⚡ Power Consumption Prediction</h1>
        <p>
            An end-to-end data science capstone project covering data cleaning,
            exploratory analysis, feature engineering, regression modeling,
            time series forecasting, and interactive deployment with Streamlit.
        </p>
    </div>
    """, unsafe_allow_html=True)

    section_header("Project Description")
    st.markdown("""
    <div class="info-strip">
        This project analyzes the <strong>UCI Individual Household Electric Power Consumption</strong>
        dataset containing over 2 million minute-level readings spanning approximately 4 years
        (Dec 2006 — Nov 2010). The goal is to accurately predict <strong>Global Active Power</strong>
        consumption using both supervised regression and time series forecasting approaches.
    </div>
    """, unsafe_allow_html=True)

    section_header("Tools & Technologies")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="insight-card">
            <h4>Languages & Core</h4>
            <ul>
                <li>Python 3.x</li>
                <li>Jupyter Notebook</li>
                <li>Streamlit</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="insight-card">
            <h4>ML & Analytics</h4>
            <ul>
                <li>Scikit-Learn</li>
                <li>XGBoost</li>
                <li>LightGBM</li>
                <li>CatBoost</li>
                <li>Statsmodels</li>
                <li>Prophet</li>
                <li>pmdarima</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="insight-card">
            <h4>Data & Visualization</h4>
            <ul>
                <li>Pandas</li>
                <li>NumPy</li>
                <li>Matplotlib</li>
                <li>Seaborn</li>
                <li>Joblib</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    section_header("Notebooks")
    notebooks = [
        ("01", "Data Cleaning", "Handling missing values, type conversion, and outlier treatment."),
        ("02", "Exploratory Data Analysis", "Temporal patterns, distributions, and correlations."),
        ("03", "Feature Engineering", "Lag, rolling, cyclical, and calendar features."),
        ("04", "Regression Models", "LR, DT, RF, XGB, LGBM, CatBoost with GridSearchCV."),
        ("05", "Time Series Forecasting", "Manual ARIMA, Auto ARIMA, SARIMA, Prophet."),
        ("06", "Final Evaluation", "Model comparison, insights, and conclusions."),
    ]
    for num, title, desc in notebooks:
        st.markdown(f"""
        <div class="info-strip">
            <strong>{num}.</strong> <strong>{title}</strong> — {desc}
        </div>
        """, unsafe_allow_html=True)

    section_header("Author")
    st.markdown("""
    <div class="insight-card" style="text-align:center;">
        <h4>Project Author</h4>
        <p style="font-size:1.5rem; font-weight:800; color:#58a6ff; margin-bottom:0.25rem;">Saikat Sarkar</p>
        <p style="font-size:0.95rem; color:#e6edf3; margin-bottom:0.15rem;">B.S. Data Science & AI</p>
        <p style="font-size:0.9rem; color:#8b949e; margin-bottom:0.75rem;">IIT Jodhpur</p>
        <p>
            <a href="https://github.com/Saikat1462/" target="_blank" style="color:#58a6ff; text-decoration:none; margin-right:1.5rem;">
                GitHub →
            </a>
            <a href="https://www.linkedin.com/in/saikat-sarkar-17151a3b1/" target="_blank" style="color:#58a6ff; text-decoration:none;">
                LinkedIn →
            </a>
        </p>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# 8. STREAMLIT APP CONFIG & ROUTING
# ──────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Power Consumption Dashboard",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_custom_css()

    # ── Sidebar Navigation ──
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding:1rem 0 0.5rem;">
            <span style="font-size:2.2rem;">⚡</span>
            <h3 style="margin:0.3rem 0 0; font-weight:800; font-size:1rem;
                        background:linear-gradient(90deg,#58a6ff,#79c0ff);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                Power Dashboard
            </h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        page = st.radio(
            "Navigation",
            options=[
                "Home",
                "Dataset Overview",
                "Exploratory Data Analysis",
                "Regression Prediction",
                "Time Series Forecasting",
                "Model Comparison",
                "Insights",
                "About",
            ],
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("---")

        st.markdown("""
        <div style="text-align:center; padding:0.5rem 0;">
            <span style="font-size:0.7rem; color:#8b949e;">
                v1.0 · Capstone Project<br>
                ML + Time Series Forecasting<br><br>
                <span style="color:#58a6ff; font-weight:600;">Saikat Sarkar</span><br>
                IIT Jodhpur
            </span>
        </div>
        """, unsafe_allow_html=True)

    # ── Page Router ──
    pages = {
        "Home": page_home,
        "Dataset Overview": page_dataset,
        "Exploratory Data Analysis": page_eda,
        "Regression Prediction": page_regression,
        "Time Series Forecasting": page_forecasting,
        "Model Comparison": page_comparison,
        "Insights": page_insights,
        "About": page_about,
    }

    pages[page]()


if __name__ == "__main__":
    main()
