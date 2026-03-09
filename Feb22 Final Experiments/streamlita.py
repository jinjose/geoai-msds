import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pickle
import os

# ==========================================
# PAGE SETUP
# ==========================================
st.set_page_config(page_title="2025 GEOAI Yield Intelligence Hub", layout="wide")

# ==========================================
# PATHS
# ==========================================
BASE_PATH = "Feb22 Final Experiments/inference-dataset/intermediate"
FEATURE_DIR = "Feb22 Final Experiments/inference-dataset/features_frozen"

# ==========================================
# MODEL CONFIG
# ==========================================
MODEL_CONFIG = {
    "Planting–Vegetative Season Model (Jun 01)": {
        "file": "Feb22 Final Experiments/exported_models/jun01/LightGBM-limited_withstorm/model.pkl",
        "cutoff": "jun01"
    },
    "Reproductive Season Model (Jul 01)": {
        "file": "Feb22 Final Experiments/exported_models/jul01/LightGBM-limited_withstorm/model.pkl",
        "cutoff": "jul01"
    },
    "Pre-Harvest Season Model (Aug 01)": {
        "file": "Feb22 Final Experiments/exported_models/aug01/LightGBM-limited_withstorm/model.pkl",
        "cutoff": "aug01"
    }
}

# ==========================================
# CUTOFF DATE LABELS
# ==========================================
CUTOFF_DATES = {
    "jun01": "June 01, 2025",
    "jul01": "July 01, 2025",
    "aug01": "August 01, 2025"
}

# ==========================================
# LOAD MODELS
# ==========================================
@st.cache_resource
def load_models():
    models = {}
    for name, cfg in MODEL_CONFIG.items():
        path = cfg["file"]
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
    return models

# ==========================================
# LOAD DATA
# ==========================================
@st.cache_data
def load_temporal_data(cutoff):

    ndvi = pd.read_csv(os.path.join(BASE_PATH, f"ndvi_until_{cutoff}.csv"))
    wx = pd.read_csv(os.path.join(BASE_PATH, f"weather_until_{cutoff}.csv"))
    storm = pd.read_csv(os.path.join(BASE_PATH, f"storm_until_{cutoff}.csv"))
    h_yield = pd.read_csv(os.path.join(BASE_PATH, "yield_history.csv"))
    augset = pd.read_csv(os.path.join(FEATURE_DIR, f"features_{cutoff}.csv"))

    ndvi["date"] = pd.to_datetime(ndvi["date"])
    wx["date"] = pd.to_datetime(wx["date"])

    t_col = "datetime" if "datetime" in storm.columns else "time"
    storm[t_col] = pd.to_datetime(storm[t_col])

    return ndvi, wx, storm, h_yield, augset


models_dict = load_models()

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.title("2025 GEO AI Annual Yield Prediction")

st.sidebar.caption(
"This system predicts county-level annual corn yield using satellite vegetation data, "
"weather observations, and historical yield records."
)

selected_stage = st.sidebar.radio(
    "Select Forecast Stage",
    list(MODEL_CONFIG.keys())
)

current_cutoff = MODEL_CONFIG[selected_stage]["cutoff"]

ndvi_all, wx_all, storm_all, h_yield_all, augset_all = load_temporal_data(current_cutoff)

all_counties = sorted(ndvi_all["county"].unique())
default_county = "benton" if "benton" in all_counties else all_counties[0]

COUNTY = st.sidebar.selectbox(
    "Select County",
    all_counties,
    index=all_counties.index(default_county)
)

# ==========================================
# FILTER COUNTY DATA
# ==========================================
ndvi_c = ndvi_all[ndvi_all["county"] == COUNTY]
wx_c = wx_all[wx_all["county"] == COUNTY]
storm_c = storm_all[storm_all["county"] == COUNTY]
yield_c = h_yield_all[h_yield_all["county"] == COUNTY]
augset_c = augset_all[augset_all["county"].astype(str).str.lower() == COUNTY.lower()]

# ==========================================
# PAGE TITLE
# ==========================================
st.title("2025 GEOAI Yield Intelligence Hub")

st.caption(
"This dashboard demonstrates an AI-driven crop yield forecasting system that "
"uses satellite vegetation signals, weather conditions, and historical yield data "
"to estimate annual corn yield during different stages of the growing season."
)

# ==========================================
# TABS
# ==========================================
tab1, tab2, tab3 = st.tabs([
    "Annual County-wise Yield Forecast",
    "Data Explorer",
    "Farmer Production/Revenue Calculator"
])

# =====================================================
# TAB 1 — YIELD FORECAST
# =====================================================
with tab1:

    st.subheader(f"Annual Yield Forecast - {selected_stage}")

    st.caption(
        "The model predicts expected annual corn yield based on environmental conditions "
        "observed up to the selected seasonal cutoff date."
    )

    model = models_dict.get(selected_stage)

    FEATURES = [
        "county",
        "rolling_3yr_mean",
        "ndvi_peak",
        "ndvi_slope",
        "temp_anomaly",
        "net_moisture_stress",
        "heat_days_gt32",
        "wind_severe_days_58_cutoff"
    ]

    X = augset_c[FEATURES].copy()
    X["county"] = X["county"].astype("category")

    num_cols = X.select_dtypes(include=["number"]).columns
    X[num_cols] = X[num_cols].fillna(0)

    if hasattr(model, "booster_"):
        pred = model.booster_.predict(X)[0]
    else:
        pred = model.predict(X)[0]

    last_actual = yield_c.sort_values("year").iloc[-1]["yield_bu_acre"]
    delta = pred - last_actual

    c1, c2 = st.columns(2)

    with c1:
        st.caption(f"Last updated: {CUTOFF_DATES[current_cutoff]}")
        st.metric(
            f"Projected Annual Yield in {COUNTY.upper()} for the year 2025",
            f"{pred:.2f} bu/ac"
        )

    with c2:
        st.caption(
            "For reference purposes only. The 2024 county yield is based on the USDA "
            "NASS survey and was officially published in 2025."
        )

        st.metric(
            f"2024 USDA NASS Reported Yield (Published 2025) — {COUNTY.upper()}",
            f"{last_actual:.2f} bu/ac",
            delta=f"{delta:+.2f} bu/ac"
        )

# =====================================================
# TAB 2 — DATA EXPLORER
# =====================================================
with tab2:

    st.subheader(
        f"Observed Crop and Weather Conditions — {COUNTY.upper()} "
        f"(through {current_cutoff.upper()})"
    )

    st.caption(
        "This chart shows observed environmental conditions used to derive model features. "
        "NDVI represents vegetation health from satellite imagery, temperature reflects weather "
        "conditions, and storm markers indicate severe wind events."
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=ndvi_c["date"],
        y=ndvi_c["NDVI"],
        name="NDVI (Vegetation Health)",
        line=dict(color="forestgreen", width=4)
    ))

    fig.add_trace(go.Scatter(
        x=wx_c["date"],
        y=wx_c["temperature"],
        name="Temperature (°C)",
        line=dict(color="orange", dash="dot"),
        yaxis="y2"
    ))

    if "wind_mph" in storm_c.columns:

        severe = storm_c[storm_c["wind_mph"] >= 58]

        if not severe.empty:

            t_col = "datetime" if "datetime" in severe.columns else "time"

            fig.add_trace(go.Scatter(
                x=severe[t_col],
                y=[ndvi_c["NDVI"].max()] * len(severe),
                mode="markers",
                name="Severe Wind ≥58mph",
                marker=dict(color="red", size=12, symbol="x")
            ))

    fig.update_layout(
        hovermode="x unified",
        yaxis=dict(title="NDVI Index"),
        yaxis2=dict(title="Temperature (°C)", overlaying="y", side="right"),
        xaxis=dict(title=f"2025 Timeline (Cutoff: {current_cutoff.upper()})")
    )

    st.plotly_chart(fig, width="stretch")

    # ==========================================
    # MODEL FEATURES
    # ==========================================
    st.subheader("Model Input Features")

    display_df = augset_c[FEATURES].copy()

    display_df = display_df.rename(columns={
        "county": "County",
        "rolling_3yr_mean": "3-Year Yield Avg",
        "ndvi_peak": "NDVI Peak",
        "ndvi_slope": "NDVI Growth Rate",
        "temp_anomaly": "Temperature Anomaly",
        "net_moisture_stress": "Net Moisture Stress",
        "heat_days_gt32": "Heat Days > 29°C",
        "wind_severe_days_58_cutoff": "Severe Wind Days (≥58 mph)"
    })

    st.dataframe(display_df, width="stretch")

    # ==========================================
    # RAW DATA
    # ==========================================
    st.subheader("Raw Data Explorer")

    dataset_choice = st.radio(
        "Inspect Dataset",
        ["NDVI", "Weather", "Storm", "History"],
        horizontal=True
    )

    if dataset_choice == "NDVI":
        st.dataframe(ndvi_c, width="stretch")

    elif dataset_choice == "Weather":
        st.dataframe(wx_c, width="stretch")

    elif dataset_choice == "Storm":
        st.dataframe(storm_c, width="stretch")

    elif dataset_choice == "History":
        st.dataframe(
            yield_c.sort_values("year", ascending=False),
            width="stretch"
        )

# =====================================================
# TAB 3 — FARMER CALCULATOR
# =====================================================
with tab3:

    st.subheader("Farm-Level Production and Revenue Estimate")

    st.caption(
        "This tool converts the AI yield forecast into estimated production and "
        "potential farm revenue based on farm size and corn market price."
    )

    st.write(f"GEOAI Model Predicted Yield: **{pred:.2f} bu/ac**")

    DEFAULT_CORN_PRICE = 4.10

    c1, c2 = st.columns(2)

    with c1:
        acres = st.number_input("Farm Size (acres)", min_value=1, value=30)

    with c2:
        corn_price = st.number_input(
            "Corn Price ($ / bushel)",
            min_value=0.0,
            value=DEFAULT_CORN_PRICE,
            step=0.10
        )

    total_bushels = acres * pred
    estimated_revenue = total_bushels * corn_price

    r1, r2 = st.columns(2)

    with r1:
        st.metric("Estimated Production", f"{total_bushels:,.0f} bushels")

    with r2:
        st.metric("Estimated Revenue", f"${estimated_revenue:,.0f}")
