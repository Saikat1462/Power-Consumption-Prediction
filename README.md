<div align="center">

# ⚡ Household Electric Power Consumption — Prediction & Forecasting

### An End-to-End Machine Learning & Time Series Capstone Project

[![Python](https://img.shields.io/badge/Python-3.13.9-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live_Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://power-consumption-prediction-a8jfcxgw9cj9huqomg9god.streamlit.app/)
[![CatBoost](https://img.shields.io/badge/CatBoost-R²_0.9715-FFCC00?style=for-the-badge&logo=catboost&logoColor=black)](https://catboost.ai)
[![Prophet](https://img.shields.io/badge/Prophet-RMSE_0.6388-0467DF?style=for-the-badge&logo=meta&logoColor=white)](https://facebook.github.io/prophet/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br>

> **Predicting household energy consumption at minute-level granularity using 6 regression models & 4 time series models, deployed via an interactive Streamlit analytics dashboard.**

<br>

[🚀 Live Dashboard](https://power-consumption-prediction-a8jfcxgw9cj9huqomg9god.streamlit.app/) · [📊 Results](#-model-performance) · [📓 Notebooks](#-project-notebooks) · [🏗️ Architecture](#%EF%B8%8F-project-architecture)

</div>

---

## 📌 Table of Contents

- [Highlights](#-highlights)
- [About the Project](#-about-the-project)
- [Dataset](#-dataset)
- [Project Architecture](#%EF%B8%8F-project-architecture)
- [Project Notebooks](#-project-notebooks)
- [Feature Engineering](#-feature-engineering)
- [Model Performance](#-model-performance)
- [Key Insights](#-key-insights)
- [Dashboard Preview](#-interactive-dashboard)
- [Quick Start](#-quick-start)
- [Tech Stack](#-tech-stack)
- [Repository Structure](#-repository-structure)
- [Future Scope](#-future-scope)
- [Author](#-author)

---

## 🔥 Highlights

<table>
<tr>
<td align="center"><b>2,075,259</b><br><sub>Minute-level Records</sub></td>
<td align="center"><b>28</b><br><sub>Engineered Features</sub></td>
<td align="center"><b>6</b><br><sub>Regression Models</sub></td>
<td align="center"><b>4</b><br><sub>Forecasting Models</sub></td>
</tr>
<tr>
<td align="center"><b>0.9715</b><br><sub>Best R² (CatBoost)</sub></td>
<td align="center"><b>0.1480</b><br><sub>Best RMSE (CatBoost)</sub></td>
<td align="center"><b>0.6388</b><br><sub>Best TS RMSE (Prophet)</sub></td>
<td align="center"><b>1385 LOC</b><br><sub>Dashboard App</sub></td>
</tr>
</table>

---

## 📖 About the Project

This capstone project tackles the challenge of **predicting and forecasting household electric power consumption** using the [UCI Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) dataset — one of the most widely used benchmarks in energy analytics.

The project follows a complete data science lifecycle:

```
Data Cleaning → EDA → Feature Engineering → Regression Modeling → Time Series Forecasting → Evaluation & Deployment
```

### 🎯 Objectives

1. **Clean & pre-process** ~2M+ minute-level power readings (Dec 2006 – Nov 2010)
2. **Explore temporal patterns** — hourly, daily, weekly, monthly, and seasonal trends
3. **Engineer 28 features** including lag values, rolling statistics, and cyclical encodings
4. **Train & tune 6 regression models** to predict Global Active Power from engineered features
5. **Apply 4 time series models** (ARIMA, SARIMA, Prophet) for pure temporal forecasting
6. **Compare all models** and extract actionable business insights
7. **Deploy** a production-quality interactive Streamlit dashboard

---

## 📊 Dataset

| Property | Detail |
|---|---|
| **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) |
| **Period** | December 2006 — November 2010 (~4 years) |
| **Granularity** | 1-minute sampling rate |
| **Total Records** | 2,075,259 |
| **Target Variable** | `Global_active_power` (kilowatt) |

### Original Features

| Feature | Description | Unit |
|---|---|---|
| `Global_active_power` | Household global active power | kW |
| `Global_reactive_power` | Household global reactive power | kW |
| `Voltage` | Average voltage | Volt |
| `Global_intensity` | Average current intensity | Ampere |
| `Sub_metering_1` | Kitchen (dishwasher, oven, microwave) | Wh |
| `Sub_metering_2` | Laundry (washing machine, dryer, fridge, light) | Wh |
| `Sub_metering_3` | Electric water heater & air conditioner | Wh |

---

## 🏗️ Project Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          RAW UCI DATASET                                 │
│                    (2M+ minute-level readings)                           │
└──────────────────────┬───────────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  01_DataCleaning.ipynb                                                   │
│  ├── Missing value imputation                                            │
│  ├── Data type conversions                                               │
│  ├── Outlier detection & treatment                                       │
│  └── Export → cleaned_power.parquet                                      │
└──────────────────────┬───────────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  02_EDA.ipynb                                                            │
│  ├── Distribution analysis                                               │
│  ├── Temporal trend exploration (hourly/daily/monthly/seasonal)          │
│  ├── Correlation heatmaps                                                │
│  └── Rolling statistics visualization                                    │
└──────────────────────┬───────────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  03_Feature_Engineering.ipynb                                            │
│  ├── Temporal features (hour, day, month, quarter, etc.)                │
│  ├── Cyclical encodings (sin/cos for hour, day_of_week, month)          │
│  ├── Lag features (1-min, 60-min, 1440-min)                             │
│  ├── Rolling window statistics (mean, std — 1hr, 24hr)                  │
│  ├── Weekend/weekday indicator                                           │
│  └── Export → featured_power.parquet                                     │
└──────────────────────┬───────────────────────────────────────────────────┘
                       │
              ┌────────┴────────┐
              ▼                 ▼
┌─────────────────────┐  ┌─────────────────────────┐
│ 04_Model_Building   │  │ 05_TimeSeries_Forecast  │
│                     │  │                         │
│ • Linear Regression │  │ • Manual ARIMA          │
│ • Decision Tree     │  │ • Auto ARIMA (pmdarima) │
│ • Random Forest     │  │ • Manual SARIMA         │
│ • XGBoost           │  │ • Prophet               │
│ • LightGBM          │  │                         │
│ • CatBoost          │  │ Saved → .pkl files      │
│                     │  │                         │
│ GridSearchCV tuning │  └────────────┬────────────┘
│ Saved → .pkl files  │               │
└──────────┬──────────┘               │
           │                          │
           └────────────┬─────────────┘
                        ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  06_Final_Conclusion.ipynb                                               │
│  ├── Side-by-side model comparison                                       │
│  ├── Performance metrics analysis                                        │
│  ├── Business insights & recommendations                                │
│  └── Limitations & future work                                           │
└──────────────────────┬───────────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  app.py — Streamlit Dashboard (1385 lines)                               │
│  ├── 🏠 Home — Project overview & quick stats                           │
│  ├── 📊 Dataset Overview — Cleaned & featured data exploration          │
│  ├── 📈 EDA — 9 interactive visualizations                              │
│  ├── 🏆 Model Comparison — Side-by-side metrics & charts               │
│  ├── 🔮 Regression Prediction — Real-time CatBoost inference            │
│  ├── 📉 Time Series Forecasting — Prophet/ARIMA/SARIMA forecasts       │
│  ├── 💡 Insights — Key findings & business recommendations             │
│  └── ℹ️  About — Author & project info                                  │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 📓 Project Notebooks

| # | Notebook | Description |
|---|---|---|
| 01 | [`01_DataCleaning.ipynb`](01_DataCleaning.ipynb) | Missing value handling, type conversion, outlier treatment, data export |
| 02 | [`02_EDA.ipynb`](02_EDA.ipynb) | Temporal patterns, distributions, correlations, rolling statistics |
| 03 | [`03_Feature_Engineering.ipynb`](03_Feature_Engineering.ipynb) | Lag features, rolling stats, cyclical encodings, calendar features |
| 04 | [`04_Model_Building.ipynb`](04_Model_Building.ipynb) | 6 regression models with GridSearchCV hyperparameter tuning |
| 05 | [`05_TimeSeries_Forecasting.ipynb`](05_TimeSeries_Forecasting.ipynb) | Manual ARIMA, Auto ARIMA, SARIMA, and Prophet |
| 06 | [`06_Final_Conclusion.ipynb`](06_Final_Conclusion.ipynb) | Model comparison, insights, recommendations, and conclusions |

---

## ⚙️ Feature Engineering

A total of **28 features** were engineered from the raw data to capture complex temporal dependencies:

### Temporal Features
| Feature | Description |
|---|---|
| `hour`, `day`, `month`, `quarter` | Calendar components extracted from datetime index |
| `day_of_week` | Day of week (0 = Monday … 6 = Sunday) |
| `dayofyear`, `weekofyear` | Position within the year |
| `is_weekend` | Binary indicator for Saturday/Sunday |

### Cyclical Encodings (sin/cos)
| Feature | Period | Purpose |
|---|---|---|
| `hour_sin`, `hour_cos` | 24 | Captures circular nature of hours |
| `day_sin`, `day_cos` | 7 | Captures weekly periodicity |
| `month_sin`, `month_cos` | 12 | Captures annual seasonality |

### Lag Features
| Feature | Lag | Rationale |
|---|---|---|
| `lag_1` | 1 minute | Immediate previous consumption |
| `lag_60` | 60 minutes | Hourly pattern capture |
| `lag_1440` | 1440 minutes (1 day) | Daily pattern capture |

### Rolling Window Statistics
| Feature | Window | Statistic |
|---|---|---|
| `previous_hour_avg` | 60 min | Rolling mean |
| `previous_hour_std` | 60 min | Rolling standard deviation |
| `previous_day_avg` | 1440 min | Rolling mean |
| `previous_day_std` | 1440 min | Rolling standard deviation |

---

## 📈 Model Performance

### 🔶 Regression Models

> **Task:** Predict `Global_active_power` using 26 engineered features (all except `Global_intensity`).

| Model | MAE | RMSE | R² | Verdict |
|---|:---:|:---:|:---:|---|
| Linear Regression | 0.7102 | 0.8826 | -0.0135 | ❌ Failed — confirms non-linearity |
| Decision Tree | 0.0663 | 0.1738 | 0.9607 | ✅ Solid baseline |
| Random Forest | 0.0602 | 0.1551 | 0.9687 | ✅ Strong ensemble |
| XGBoost | 0.0629 | 0.1562 | 0.9682 | ✅ Competitive |
| LightGBM | 0.0631 | 0.1497 | 0.9708 | ✅ Runner-up |
| **CatBoost** | **0.0639** | **0.1480** | **0.9715** | 🏆 **Best Model** |

### 🔷 Time Series Forecasting Models

> **Task:** Forecast future `Global_active_power` using only historical temporal patterns.

| Model | AIC | MAE | RMSE | MAPE (%) | Verdict |
|---|:---:|:---:|:---:|:---:|---|
| Manual ARIMA | 51,775.82 | 0.5775 | 0.7011 | 99.62 | ⚠️ Baseline |
| Auto ARIMA | 53,974.67 | 0.6241 | 0.7364 | 108.58 | ⚠️ Below manual |
| Manual SARIMA | 48,235.51 | 0.5065 | 0.6832 | 70.93 | ✅ Best ARIMA variant |
| **Prophet** | — | **0.4884** | **0.6388** | **73.37** | 🏆 **Best Model** |

> [!NOTE]
> **Manual SARIMA model file (`manual_sarima.pkl`) is NOT included in this repository** as it is approximately **~14 GB** in size, which exceeds GitHub's file size limits. To use the SARIMA model, you will need to retrain it locally by running [`05_TimeSeries_Forecasting.ipynb`](05_TimeSeries_Forecasting.ipynb). All other model files are included in the repo.

---

## 💡 Key Insights

### 🌟 Model Performance

- **CatBoost** is the regression champion with R² = **0.9715** — it explains 97.15% of the variance in power consumption
- **Prophet** leads forecasting with the lowest RMSE (**0.6388**) and MAE (**0.4884**)
- **Linear Regression completely failed** (negative R²), confirming that power consumption has strong **non-linear** relationships
- All tree-based ensembles (DT, RF, XGBoost, LightGBM, CatBoost) significantly outperformed Linear Regression
- **Seasonal modeling is critical** — SARIMA outperformed both ARIMA variants thanks to its seasonal component

### 📊 Temporal Patterns

- ⏰ **Peak Hours:** Power consumption peaks during evening hours (**17:00–21:00**) — prime demand planning window
- ❄️ **Seasonal:** Winter months show significantly higher consumption — heating drives demand
- 📅 **Weekend Effect:** Weekends exhibit different usage patterns with higher mid-day consumption
- 📉 **Summer Lows:** Summer consistently shows the lowest average power consumption

### 🧠 Feature Engineering Impact

- Regression models (R² up to **0.97**) vastly outperformed time series models (RMSE ~0.64)
- This proves that **feature engineering** (lag values, rolling stats, cyclical encodings) is a **game-changer** for power consumption prediction
- Pure time series models rely solely on historical patterns without external features, limiting their performance ceiling

### 🏢 Business Recommendations

- **Demand Planning:** Use evening peak patterns (17:00–21:00) for load balancing
- **Anomaly Detection:** Rolling statistics (mean & std) serve as excellent detectors for unusual consumption spikes
- **Deployment:** CatBoost with feature-engineered inputs is recommended for real-time prediction APIs
- **Energy Savings:** Target high-consumption appliances (Sub_metering_3: water heater & AC) during peak seasons

---

## 🖥️ Interactive Dashboard

### 🌐 [**➡️ Try the Live Dashboard Here**](https://power-consumption-prediction-a8jfcxgw9cj9huqomg9god.streamlit.app/)

The project includes a **production-quality Streamlit dashboard** (`app.py` — 1,385 lines) with 8 interactive pages:

| Page | Feature |
|---|---|
| 🏠 **Home** | Project overview, quick stats, workflow visualization |
| 📊 **Dataset Overview** | Explore cleaned & feature-engineered datasets with statistics |
| 📈 **EDA** | 9 interactive visualizations — distributions, trends, correlations |
| 🏆 **Model Comparison** | Side-by-side regression & forecasting metrics with interactive charts |
| 🔮 **Regression Prediction** | Enter custom feature values → get real-time CatBoost predictions |
| 📉 **Time Series Forecasting** | Generate Prophet/ARIMA/SARIMA forecasts for custom horizons |
| 💡 **Insights** | Key findings, business recommendations, and limitations |
| ℹ️ **About** | Project description, tools, notebooks, and author info |

### Dashboard Features
- 🎨 **Custom dark theme** with polished CSS (glassmorphism, gradients, animations)
- ⚡ **Cached data & model loading** for fast performance
- 📱 **Responsive layout** with wide-mode support
- 📥 **CSV download** for forecast results
- 🔄 **Interactive forms** for prediction input

---

## 🚀 Quick Start

### ☁️ Try it Online (No Installation Needed!)

> **[👉 Launch the Live Dashboard](https://power-consumption-prediction-a8jfcxgw9cj9huqomg9god.streamlit.app/)** — explore the full app instantly in your browser.

### 🖥️ Run Locally

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Saikat1462/Power-Consumption-Prediction.git
cd Power-Consumption-Prediction

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Run the Dashboard

```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501` 🚀

### Run Notebooks

Open the notebooks sequentially in Jupyter:

```bash
jupyter notebook
```

Navigate to notebooks `01` through `06` in order.

---

## 🛠️ Tech Stack

<table>
<tr>
<td align="center" width="33%">

**Languages & Core**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)

</td>
<td align="center" width="33%">

**ML & Analytics**

![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189FDD?style=flat-square&logo=xgboost&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-02569B?style=flat-square)
![CatBoost](https://img.shields.io/badge/CatBoost-FFCC00?style=flat-square)
![Prophet](https://img.shields.io/badge/Prophet-0467DF?style=flat-square&logo=meta&logoColor=white)
![Statsmodels](https://img.shields.io/badge/Statsmodels-4051B5?style=flat-square)
![pmdarima](https://img.shields.io/badge/pmdarima-6DB33F?style=flat-square)

</td>
<td align="center" width="33%">

**Data & Visualization**

![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square)
![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=flat-square)
![Joblib](https://img.shields.io/badge/Joblib-E34A86?style=flat-square)

</td>
</tr>
</table>

---

## 📁 Repository Structure

```
Power-Consumption-Prediction/
│
├── 📓 01_DataCleaning.ipynb          # Data cleaning & preprocessing
├── 📓 02_EDA.ipynb                    # Exploratory Data Analysis
├── 📓 03_Feature_Engineering.ipynb    # Feature creation pipeline
├── 📓 04_Model_Building.ipynb         # Regression model training & tuning
├── 📓 05_TimeSeries_Forecasting.ipynb # ARIMA, SARIMA, Prophet
├── 📓 06_Final_Conclusion.ipynb       # Model comparison & conclusions
│
├── ⚡ app.py                          # Streamlit dashboard (1385 lines)
│
├── 📦 cleaned_power.parquet           # Cleaned dataset
├── 📦 featured_power.parquet          # Feature-engineered dataset
│
├── 🤖 best_model_lr.pkl              # Linear Regression model
├── 🤖 best_model_dt.pkl              # Decision Tree model
├── 🤖 best_model_rf.pkl              # Random Forest model
├── 🤖 best_model_xgb.pkl             # XGBoost model
├── 🤖 best_model_lgb.pkl             # LightGBM model
├── 🤖 best_model_cat.pkl             # CatBoost model (Best Regression)
│
├── 📈 prophet.pkl                     # Prophet model (Best Forecasting)
├── 📈 auto_arima.pkl                  # Auto ARIMA model
├── 📈 manual_arima.pkl                # Manual ARIMA model
├── 📈 manual_sarima.pkl               # ⚠️ NOT on GitHub (~14 GB) — retrain locally
│
├── 📋 requirements.txt                # Python dependencies
├── ⚙️ .streamlit/config.toml         # Streamlit dark theme config
└── 📖 README.md                       # This file
```

---

## 🔮 Future Scope

- 🧠 **Deep Learning:** Implement LSTM and Transformer-based models for sequence prediction
- 🌤️ **External Data:** Integrate weather API data (temperature, humidity) for richer features
- ⚙️ **Hyperparameter Optimization:** Use Bayesian optimization (Optuna) for more efficient tuning
- 🚀 **API Deployment:** Build a FastAPI endpoint for real-time CatBoost predictions
- 📊 **Multi-step Forecasting:** Extend to multi-horizon recursive forecasting
- 🏠 **Smart Home Integration:** Connect with IoT devices for live consumption monitoring

---

## 👨‍💻 Author

<div align="center">

### **Saikat Sarkar**

**B.S. Data Science & AI** · **IIT Jodhpur**

[![GitHub](https://img.shields.io/badge/GitHub-Saikat1462-181717?style=for-the-badge&logo=github)](https://github.com/Saikat1462)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Saikat_Sarkar-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/saikat-sarkar-17151a3b1/)
[![Live App](https://img.shields.io/badge/Live_App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://power-consumption-prediction-a8jfcxgw9cj9huqomg9god.streamlit.app/)

</div>

---

<div align="center">

**⭐ If you found this project helpful, please give it a star!**

Made with ❤️ and lots of ☕ by **Saikat Sarkar** at **IIT Jodhpur**

</div>
